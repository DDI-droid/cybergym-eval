from __future__ import annotations

import os
import secrets
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import docker
from docker.errors import NotFound

from .changes import record_change
from .prepare import INTERNAL_NETWORK

SECRETS_DIR = Path.home() / ".secrets"
API_KEY_FILE = SECRETS_DIR / "cybergym_api_key"


def server_dir(config: dict[str, Any]) -> Path:
    return Path(config["work_root"]) / "server"


def server_host(config: dict[str, Any]) -> str:
    network = INTERNAL_NETWORK if config["use_firewall"] else "bridge"
    try:
        return docker.from_env().networks.get(network).attrs["IPAM"]["Config"][0]["Gateway"]
    except NotFound:
        raise RuntimeError(f"Docker network {network} not found; run prepare first") from None


def server_url(config: dict[str, Any]) -> str:
    return f"http://{server_host(config)}:{config['server_port']}"


def api_key() -> str:
    if not API_KEY_FILE.exists():
        SECRETS_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        API_KEY_FILE.write_text(f"cybergym-{secrets.token_hex(16)}\n", encoding="utf-8")
        API_KEY_FILE.chmod(0o600)
    return API_KEY_FILE.read_text(encoding="utf-8").strip()


def _pid_file(config: dict[str, Any]) -> Path:
    return server_dir(config) / "server.pid"


def _is_our_server(pid: int) -> bool:
    proc = Path("/proc") / str(pid)
    try:
        cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        owner = proc.stat().st_uid
    except OSError:
        return False
    return owner == os.getuid() and "cybergym.server" in cmdline


def _running_pid(config: dict[str, Any]) -> int | None:
    path = _pid_file(config)
    if not path.exists():
        return None
    pid = int(path.read_text(encoding="utf-8").strip())
    return pid if _is_our_server(pid) else None


def _listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def start_server(config: dict[str, Any]) -> str:
    host, port = server_host(config), config["server_port"]
    url = f"http://{host}:{port}"
    if _running_pid(config):
        return url
    if _listening(host, port):
        raise RuntimeError(f"Something else is already listening on {host}:{port}")
    directory = server_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    root = Path(config["benchmark_root"])
    cmd = [
        str(root / ".venv" / "bin" / "python"), "-m", "cybergym.server",
        "--host", host,
        "--port", str(port),
        "--mask_map_path", str(root / "mask_map.json"),
        "--log_dir", str(directory / "logs"),
        "--db_path", str(directory / "poc.db"),
        "--binary_dir", config["binary_dir"],
    ]
    env = {**os.environ, "CYBERGYM_API_KEY": api_key()}
    with (directory / "server.out").open("ab") as out:
        process = subprocess.Popen(cmd, cwd=root, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    _pid_file(config).write_text(str(process.pid), encoding="utf-8")
    record_change("server_pid_files", str(_pid_file(config)))
    for _ in range(60):
        if _listening(host, port):
            return url
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with code {process.returncode}; see {directory / 'server.out'}")
        time.sleep(0.5)
    raise RuntimeError(f"Server did not start listening on {url} within 30 s")


def stop_server(config: dict[str, Any]) -> bool:
    pid = _running_pid(config)
    if pid is None:
        _pid_file(config).unlink(missing_ok=True)
        return False
    os.killpg(pid, signal.SIGTERM)
    for _ in range(20):
        if not _is_our_server(pid):
            break
        time.sleep(0.5)
    else:
        os.killpg(pid, signal.SIGKILL)
    _pid_file(config).unlink(missing_ok=True)
    return True


def _binary_dir_of(pid: int | None) -> str | None:
    if pid is None:
        return None
    try:
        args = (Path("/proc") / str(pid) / "cmdline").read_bytes().decode(errors="replace").split("\0")
    except OSError:
        return None
    return args[args.index("--binary_dir") + 1] if "--binary_dir" in args[:-1] else None


def server_status(config: dict[str, Any]) -> dict[str, Any]:
    try:
        host = server_host(config)
    except RuntimeError as error:
        return {"url": None, "pid": None, "listening": False, "binary_dir": None, "detail": str(error)}
    port = config["server_port"]
    pid = _running_pid(config)
    return {
        "url": f"http://{host}:{port}",
        "pid": pid,
        "listening": _listening(host, port),
        "binary_dir": _binary_dir_of(pid),
        "detail": str(server_dir(config)),
    }
