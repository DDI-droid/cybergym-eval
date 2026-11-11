from __future__ import annotations

import http.client
import json
import os
import secrets
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .changes import record_change

SECRETS_DIR = Path.home() / ".secrets"
MASTER_KEY_FILE = SECRETS_DIR / "cybergym_shim_master_key"
RETRY_STATUSES = {429, 500, 502, 503, 504}
STREAM_IDLE_TIMEOUT_SECONDS = 90
KEEPALIVE_SECONDS = 15
REQUEST_TIMEOUT_SECONDS = 240
PROCESS_MARKER = "cybergym_eval.shim"


class Meter:
    def __init__(self, price_in: float | None, price_out: float | None, log_path: Path):
        self.price_in, self.price_out = price_in, price_out
        self.log_path = log_path
        self.keys: dict[str, dict[str, Any]] = {}
        # Counters of revoked keys. A request already in flight when its key is revoked is still attributed to it.
        self.retired: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()

    def issue(self, alias: str) -> str:
        key = "cge-" + secrets.token_hex(16)
        with self.lock:
            self.keys[key] = {
                "alias": alias, "requests": 0, "input_tokens": 0, "output_tokens": 0,
                "cost_usd": 0.0, "throttled": 0, "retries": 0, "errors": 0, "model_seconds": 0.0,
                "discarded_attempts": 0, "discarded_input_tokens": 0, "discarded_output_tokens": 0,
                "unmeasured_attempts": 0, "unmeasured_input_tokens_est": 0, "inflight": 0,
            }
        return key

    def _counters(self, key: str) -> dict[str, Any] | None:
        """Counters of a live key, or of a revoked one (caller holds the lock)."""
        return self.keys.get(key) or self.retired.get(key)

    def admission(self, key: str) -> tuple[int, str, str] | None:
        """None if a new upstream attempt may start for this key, else (HTTP status, error type, message)."""
        with self.lock:
            counters = self.keys.get(key)
            if counters is None:
                if key in self.retired:
                    return 403, "key_revoked", "this execution has ended; its key was revoked"
                return 401, "invalid_request_error", "unknown key"
            if counters.get("blocked"):
                return 403, "time_limit", "this execution's time limit was reached"
        return None

    def begin(self, key: str) -> None:
        with self.lock:
            counters = self._counters(key)
            if counters is not None:
                counters["inflight"] += 1

    def end(self, key: str) -> None:
        with self.lock:
            counters = self._counters(key)
            if counters is not None and counters["inflight"] > 0:
                counters["inflight"] -= 1

    def retire(self, key: str) -> bool:
        """Revoke a key. Its counters stay readable and keep receiving the usage of requests in flight."""
        with self.lock:
            counters = self.keys.pop(key, None)
            if counters is None:
                return False
            counters["revoked"] = True
            self.retired[key] = counters
            return True

    def info(self, key: str) -> dict[str, Any] | None:
        with self.lock:
            counters = self._counters(key)
            return dict(counters) if counters is not None else None

    def record_discarded(self, key: str, usage: dict[str, Any] | None, estimated_input: float, seconds: float, reason: str) -> None:
        inp = int((usage or {}).get("prompt_tokens") or 0)
        out = int((usage or {}).get("completion_tokens") or 0)
        measured = inp > 0 or out > 0
        cost = ((inp * (self.price_in or 0)) + (out * (self.price_out or 0))) / 1e6
        with self.lock:
            counters = self._counters(key)
            alias = None
            if counters is not None:
                alias = counters["alias"]
                counters["discarded_attempts"] += 1
                if measured:
                    counters["input_tokens"] += inp
                    counters["output_tokens"] += out
                    counters["cost_usd"] = round(counters["cost_usd"] + cost, 6)
                    counters["discarded_input_tokens"] += inp
                    counters["discarded_output_tokens"] += out
                else:
                    counters["unmeasured_attempts"] += 1
                    counters["unmeasured_input_tokens_est"] += int(estimated_input)
            line = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "alias": alias, "status": 502, "discarded": True,
                "reason": reason, "measured": measured, "seconds": round(seconds, 3), "input_tokens": inp, "output_tokens": out,
                "est_input_tokens_unmeasured": 0 if measured else int(estimated_input), "throttled": 0, "retries": 0, "stream": True,
            }
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(line) + "\n")

    def record(self, key: str, status: int, usage: dict[str, Any] | None, seconds: float, throttled: int, retries: int, stream: bool) -> None:
        inp = int((usage or {}).get("prompt_tokens") or 0)
        out = int((usage or {}).get("completion_tokens") or 0)
        cost = ((inp * (self.price_in or 0)) + (out * (self.price_out or 0))) / 1e6
        with self.lock:
            counters = self._counters(key)
            if counters is not None:
                counters["requests"] += 1
                counters["input_tokens"] += inp
                counters["output_tokens"] += out
                counters["cost_usd"] = round(counters["cost_usd"] + cost, 6)
                counters["throttled"] += throttled
                counters["retries"] += retries
                counters["errors"] += status >= 400
                counters["model_seconds"] = round(counters["model_seconds"] + seconds, 3)
                alias = counters["alias"]
            else:
                alias = None
            line = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "alias": alias, "status": status,
                "seconds": round(seconds, 3), "input_tokens": inp, "output_tokens": out, "throttled": throttled,
                "retries": retries, "stream": stream,
            }
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(line) + "\n")


def usage_of(lines: list[bytes]) -> dict[str, Any] | None:
    usage = None
    for line in lines:
        if line.startswith(b"data: {"):
            try:
                usage = json.loads(line[6:]).get("usage") or usage
            except ValueError:
                pass
    return usage


def make_handler(meter: Meter, upstream: str, upstream_key: str, model: str, master_key: str, max_retries: int):
    target = urlparse(upstream)
    chat_path = target.path.rstrip("/") + "/chat/completions"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def _json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _bearer(self) -> str:
            value = self.headers.get("Authorization", "")
            return value[7:] if value.startswith("Bearer ") else ""

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                return self._json(200, {"ok": True, "model": model})
            if parsed.path.rstrip("/") == "/v1/models":
                return self._json(200, {"object": "list", "data": [{"id": model, "object": "model"}]})
            if parsed.path == "/key/info" and self._bearer() == master_key:
                info = meter.info(parse_qs(parsed.query).get("key", [""])[0])
                return self._json(200 if info else 404, {"info": info})
            return self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path in ("/key/generate", "/key/delete", "/key/block"):
                if self._bearer() != master_key:
                    return self._json(403, {"error": "forbidden"})
                body = self._body()
                if path == "/key/generate":
                    return self._json(200, {"key": meter.issue(str(body.get("alias", "")))})
                name = str(body.get("key", ""))
                if path == "/key/block":
                    with meter.lock:
                        counters = meter.keys.get(name)
                        if counters is not None:
                            counters["blocked"] = True
                    return self._json(200, {"blocked": counters is not None})
                return self._json(200, {"deleted": meter.retire(name)})
            if path.rstrip("/") != "/v1/chat/completions":
                return self._json(404, {"error": "not found"})
            key = self._bearer()
            denied = meter.admission(key)
            if denied:
                return self._json(denied[0], {"error": {"message": denied[2], "type": denied[1]}})
            body = self._body()
            body["model"] = model
            stream = bool(body.get("stream"))
            if stream:
                body["stream_options"] = {"include_usage": True}
            self._forward(key, json.dumps(body).encode(), stream)

        def _open(self, payload: bytes, stream: bool) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
            connection_class = http.client.HTTPSConnection if target.scheme == "https" else http.client.HTTPConnection
            timeout = STREAM_IDLE_TIMEOUT_SECONDS if stream else REQUEST_TIMEOUT_SECONDS
            connection = connection_class(target.hostname, target.port, timeout=timeout)
            connection.request(
                "POST", chat_path, body=payload,
                headers={"Authorization": f"Bearer {upstream_key}", "Content-Type": "application/json"},
            )
            return connection, connection.getresponse()

        def _forward(self, key: str, payload: bytes, stream: bool) -> None:
            """Relay one request upstream, retrying failures.

            Usage is metered as soon as the upstream outcome is known, before anything is written to the client,
            so a client that has gone away cannot hide what was spent. A request that ends without any outcome
            recorded is booked as an unmeasured attempt. The key is rechecked before every new upstream attempt;
            requests already in flight stay attributed to their key even if it is revoked meanwhile.
            """
            meter.begin(key)
            outcome = {"recorded": False}
            started = time.monotonic()
            try:
                self._relay(key, payload, stream, started, outcome)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the client went away; what the upstream cost was recorded before any write
            finally:
                if not outcome["recorded"]:
                    meter.record_discarded(
                        key, None, len(payload) / 3.3, time.monotonic() - started,
                        "the handler ended before the upstream outcome was recorded",
                    )
                meter.end(key)

        def _relay(self, key: str, payload: bytes, stream: bool, started: float, outcome: dict[str, bool]) -> None:
            throttled = retries = 0
            client_gone = threading.Event()

            def settle(status: int, usage: dict[str, Any] | None) -> None:
                outcome["recorded"] = True
                meter.record(key, status, usage, time.monotonic() - started, throttled, retries, stream)

            def client_closed() -> bool:
                """True once the client has closed its end (a peek at the socket; no data is consumed)."""
                connection = getattr(self, "connection", None)
                if connection is None:
                    return False
                try:
                    readable, _, _ = select.select([connection], [], [], 0)
                    return bool(readable) and connection.recv(1, socket.MSG_PEEK) == b""
                except OSError:
                    return True

            def refusal() -> tuple[int, str, str] | None:
                """Why no new upstream attempt may start: the client left, or the key is revoked or blocked."""
                if client_gone.is_set() or client_closed():
                    return 499, "client_disconnected", "the client disconnected"
                return meter.admission(key)

            for attempt in range(max_retries + 1):
                if attempt > 0:
                    denied = refusal()
                    if denied:
                        settle(denied[0], None)
                        return self._json(denied[0], {"error": {
                            "message": f"no further upstream attempt was made: {denied[2]}", "type": denied[1]}})
                try:
                    connection, response = self._open(payload, stream)
                except OSError as error:
                    if attempt < max_retries:
                        retries += 1
                        time.sleep(min(30, 2 ** attempt))
                        continue
                    settle(502, None)
                    return self._json(502, {"error": {"message": f"upstream unreachable: {error}", "type": "upstream_error"}})
                if response.status in RETRY_STATUSES and attempt < max_retries:
                    throttled += response.status == 429
                    retries += 1
                    response.read()
                    connection.close()
                    time.sleep(min(30, 2 ** attempt))
                    continue
                break
            if not stream or response.status != 200:
                data = response.read()
                connection.close()
                usage = None
                try:
                    usage = json.loads(data).get("usage")
                except (ValueError, AttributeError):
                    pass
                settle(response.status, usage)
                self.send_response(response.status)
                self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
            except OSError:
                connection.close()
                raise
            stop = threading.Event()

            def keepalive() -> None:
                while not stop.wait(KEEPALIVE_SECONDS):
                    try:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                    except OSError:
                        client_gone.set()
                        return

            pinger = threading.Thread(target=keepalive, daemon=True)
            pinger.start()
            attempt_started = time.monotonic()
            streamed = True
            try:
                lines, problem = self._read_stream(response)
                connection.close()
                while problem and attempt < max_retries:
                    if streamed:
                        meter.record_discarded(key, usage_of(lines), len(payload) / 3.3, time.monotonic() - attempt_started, problem)
                    attempt += 1
                    retries += 1
                    time.sleep(min(30, 2 ** attempt))
                    streamed = False
                    denied = refusal()
                    if denied:
                        problem = f"{problem}; no further upstream attempt was made: {denied[2]}"
                        break
                    attempt_started = time.monotonic()
                    try:
                        connection, response = self._open(payload, stream)
                    except OSError:
                        continue
                    if response.status != 200:
                        response.read()
                        connection.close()
                        continue
                    streamed = True
                    lines, problem = self._read_stream(response)
                    connection.close()
            finally:
                stop.set()
                pinger.join()
            usage = usage_of(lines) if streamed else None
            if problem:
                # Every attempt failed. Nothing an attempt streamed is forwarded: a truncated reply can hold a
                # complete-looking tool call, and the client must not act on it. Meter first, then answer with an
                # explicit terminal error event (an SDK client raises on it; there is no [DONE]).
                if streamed and usage is None:
                    meter.record_discarded(key, None, len(payload) / 3.3, time.monotonic() - attempt_started, problem)
                settle(502, usage)
                event = {"error": {
                    "message": f"upstream stream failed after {attempt + 1} attempt(s): {problem}",
                    "type": "upstream_error", "code": "upstream_stream_failed"}}
                self.close_connection = True
                self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
                self.wfile.flush()
                return
            settle(200, usage)
            self.wfile.write(b"".join(lines))
            self.wfile.flush()

        def _read_stream(self, response: http.client.HTTPResponse) -> tuple[list[bytes], str | None]:
            lines: list[bytes] = []
            finished = False
            problem = None
            try:
                while True:
                    line = response.readline()
                    if not line:
                        break
                    lines.append(line)
                    if line.startswith(b"data: [DONE]"):
                        finished = True
                    elif line.startswith(b"data: {") and b'"error"' in line:
                        try:
                            if "error" in json.loads(line[6:]):
                                problem = "error event in the stream"
                        except ValueError:
                            pass
            except (OSError, http.client.HTTPException) as error:
                problem = f"stream broke: {type(error).__name__}"
            if problem is None and not finished:
                problem = "stream ended without [DONE]"
            return lines, problem

    return Handler


def serve() -> None:
    price_in = os.environ.get("SHIM_PRICE_IN_PER_M")
    price_out = os.environ.get("SHIM_PRICE_OUT_PER_M")
    meter = Meter(float(price_in) if price_in else None, float(price_out) if price_out else None, Path(os.environ["SHIM_LOG"]))
    handler = make_handler(
        meter,
        os.environ["SHIM_UPSTREAM"],
        os.environ["SHIM_UPSTREAM_KEY"],
        os.environ["SHIM_MODEL"],
        os.environ["SHIM_MASTER_KEY"],
        int(os.environ.get("SHIM_MAX_RETRIES", "6")),
    )
    server = ThreadingHTTPServer((os.environ["SHIM_BIND"], int(os.environ["SHIM_PORT"])), handler)
    server.daemon_threads = True
    server.serve_forever()


def master_key() -> str:
    if not MASTER_KEY_FILE.exists():
        SECRETS_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        MASTER_KEY_FILE.write_text(secrets.token_hex(24) + "\n", encoding="utf-8")
        MASTER_KEY_FILE.chmod(0o600)
    return MASTER_KEY_FILE.read_text(encoding="utf-8").strip()


def shim_dir(config: dict[str, Any]) -> Path:
    return Path(config["work_root"]) / "shim"


def _pid_file(config: dict[str, Any]) -> Path:
    return shim_dir(config) / "shim.pid"


def _is_ours(pid: int) -> bool:
    proc = Path("/proc") / str(pid)
    try:
        cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        owner = proc.stat().st_uid
    except OSError:
        return False
    return owner == os.getuid() and PROCESS_MARKER in cmdline


def _running_pid(config: dict[str, Any]) -> int | None:
    path = _pid_file(config)
    if not path.exists():
        return None
    pid = int(path.read_text(encoding="utf-8").strip())
    return pid if _is_ours(pid) else None


def _listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def shim_url(config: dict[str, Any]) -> str:
    from .server import server_host

    return f"http://{server_host(config)}:{config['shim_port']}/v1"


def start_shim(config: dict[str, Any]) -> str:
    from .server import server_host

    host, port = server_host(config), config["shim_port"]
    url = f"http://{host}:{port}/v1"
    if _running_pid(config):
        return url
    if _listening(host, port):
        raise RuntimeError(f"Something else is already listening on {host}:{port}")
    upstream_key = os.getenv("OPENAI_API_KEY") or os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    if not upstream_key:
        raise RuntimeError("No model credential in the environment; set env_file in the config")
    directory = shim_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "SHIM_UPSTREAM": config["openai_base_url"],
        "SHIM_UPSTREAM_KEY": upstream_key,
        "SHIM_MODEL": config["model"],
        "SHIM_MASTER_KEY": master_key(),
        "SHIM_BIND": host,
        "SHIM_PORT": str(port),
        "SHIM_LOG": str(directory / "requests.jsonl"),
    }
    for key in ("price_in_per_m", "price_out_per_m"):
        if config.get(key) is not None:
            env[f"SHIM_{key.upper()}"] = str(config[key])
    with (directory / "shim.out").open("ab") as out:
        process = subprocess.Popen(
            [sys.executable, "-m", PROCESS_MARKER], env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True
        )
    _pid_file(config).write_text(str(process.pid), encoding="utf-8")
    record_change("server_pid_files", str(_pid_file(config)))
    for _ in range(40):
        if _listening(host, port):
            return url
        if process.poll() is not None:
            raise RuntimeError(f"Shim exited with code {process.returncode}; see {directory / 'shim.out'}")
        time.sleep(0.25)
    raise RuntimeError(f"Shim did not start listening on {url} within 10 s")


def stop_shim(config: dict[str, Any]) -> bool:
    pid = _running_pid(config)
    if pid is None:
        _pid_file(config).unlink(missing_ok=True)
        return False
    os.killpg(pid, signal.SIGTERM)
    for _ in range(20):
        if not _is_ours(pid):
            break
        time.sleep(0.25)
    else:
        os.killpg(pid, signal.SIGKILL)
    _pid_file(config).unlink(missing_ok=True)
    return True


def shim_status(config: dict[str, Any]) -> dict[str, Any]:
    from .server import server_host

    try:
        host = server_host(config)
    except RuntimeError as error:
        return {"url": None, "pid": None, "listening": False, "detail": str(error)}
    return {
        "url": f"http://{host}:{config['shim_port']}/v1",
        "pid": _running_pid(config),
        "listening": _listening(host, config["shim_port"]),
        "detail": str(shim_dir(config)),
    }


def _manage(config: dict[str, Any], method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    base = shim_url(config).removesuffix("/v1")
    request = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={"Authorization": f"Bearer {master_key()}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def issue_key(config: dict[str, Any], alias: str) -> str:
    return _manage(config, "POST", "/key/generate", {"alias": alias})["key"]


def key_usage(config: dict[str, Any], key: str) -> dict[str, Any] | None:
    return _manage(config, "GET", f"/key/info?key={key}")["info"]


def revoke_key(config: dict[str, Any], key: str) -> None:
    _manage(config, "POST", "/key/delete", {"key": key})


def block_key(config: dict[str, Any], key: str) -> None:
    _manage(config, "POST", "/key/block", {"key": key})


if __name__ == "__main__":
    serve()
