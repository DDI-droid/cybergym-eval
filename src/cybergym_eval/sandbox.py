from __future__ import annotations

import io
import json
import os
import posixpath
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import docker
from docker.errors import APIError, NotFound

from .config import RunSpec, work_id
from .prepare import INTERNAL_NETWORK
from .scheduler import EXECUTION_MARKER
from .server import server_host
from .shim import block_key

SANDBOX_IMAGE = "cybergym-harness-base:1"
MAX_OUTPUT_CHARS = 20000


class SandboxClosed(RuntimeError):
    pass


@dataclass(frozen=True)
class Task:
    id: str
    difficulty: str
    time_limit_seconds: int
    deadline: float

    def seconds_left(self) -> float:
        return self.deadline - time.time()


@dataclass(frozen=True)
class ModelAccess:
    base_url: str
    api_key: str
    model: str

    def dspy(self, **kwargs: Any) -> Any:
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        import dspy

        return dspy.LM(f"openai/{self.model}", api_base=self.base_url, api_key=self.api_key, cache=False, **kwargs)


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    output: str
    timed_out: bool

    def __str__(self) -> str:
        return self.output if self.exit_code == 0 else f"[exit code {self.exit_code}]\n{self.output}"


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    return f"{text[:half]}\n...[{len(text) - MAX_OUTPUT_CHARS} characters omitted]...\n{text[-half:]}"


def remove_sandboxes(execution_id: str) -> None:
    for container in docker.from_env().containers.list(all=True, filters={"label": f"execution={execution_id}"}):
        try:
            container.remove(force=True)
        except NotFound:
            pass


class Sandbox:
    def __init__(
        self, config: dict[str, Any], execution_id: str, workspace: Path, output: Path,
        image: str = SANDBOX_IMAGE, log_path: Path | None = None,
    ):
        self.config = config
        self.execution_id = execution_id
        self.workspace = workspace
        self.output = output
        self.image = image
        self.log_path = log_path
        self.container: Any = None
        self.oom_killed = False
        self.steps = 0
        self.commands = 0
        self.started = time.monotonic()
        self._lock = threading.Lock()

    def __enter__(self) -> Sandbox:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def start(self) -> None:
        if self.log_path is not None:
            self.log_path.touch()
        remove_sandboxes(self.execution_id)
        gateway = server_host(self.config)
        self.container = docker.from_env().containers.run(
            self.image,
            command=["sleep", "infinity"],
            name=f"cge-{self.execution_id}-{uuid.uuid4().hex[:6]}",
            detach=True,
            environment={"HOME": "/tmp/home", "NO_PROXY": gateway, "no_proxy": gateway},
            volumes={
                str(self.workspace): {"bind": "/workspace", "mode": "rw"},
                str(self.output): {"bind": "/output", "mode": "rw"},
            },
            working_dir="/workspace",
            user=f"{os.getuid()}:{os.getgid()}",
            network=INTERNAL_NETWORK if self.config["use_firewall"] else "bridge",
            nano_cpus=int(float(self.config["task_cpus"]) * 1e9),
            mem_limit=f"{self.config['task_memory_gb']}g",
            labels={"cybergym-eval": "sandbox", "cybergym-eval.work": work_id(self.config), "execution": self.execution_id},
        )
        self.container.exec_run(["mkdir", "-p", "/tmp/home"])

    def _record(self, kind: str, **fields: Any) -> None:
        with self._lock:
            self.steps += 1
            step = {"step": self.steps, "type": kind, "t": round(time.monotonic() - self.started, 1), **fields}
            if self.log_path is not None:
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(step) + "\n")

    def _live(self) -> Any:
        if self.container is None:
            raise SandboxClosed("the sandbox is closed")
        return self.container

    def run(self, command: str, timeout: int = 300, workdir: str = "/workspace") -> CommandResult:
        container = self._live()
        started = time.monotonic()
        try:
            exit_code, raw = container.exec_run(["timeout", "-k", "5", str(int(timeout)), "bash", "-lc", command], workdir=workdir)
        except (NotFound, APIError) as error:
            raise SandboxClosed(f"the sandbox is gone: {error}") from None
        text = _clip((raw or b"").decode("utf-8", errors="replace"))
        self.commands += 1
        self._record("command", command=command, exit_code=exit_code, output=text, seconds=round(time.monotonic() - started, 1))
        return CommandResult(exit_code=exit_code, output=text, timed_out=exit_code in (124, 137))

    def read(self, path: str) -> bytes:
        container = self._live()
        try:
            stream, _ = container.get_archive(path)
        except NotFound:
            raise FileNotFoundError(path) from None
        with tarfile.open(fileobj=io.BytesIO(b"".join(stream))) as archive:
            member = next((item for item in archive.getmembers() if item.isfile()), None)
            if member is None:
                raise IsADirectoryError(path)
            data = archive.extractfile(member).read()
        self._record("read", path=path, bytes=len(data))
        return data

    def write(self, path: str, data: bytes | str) -> None:
        container = self._live()
        payload = data.encode() if isinstance(data, str) else bytes(data)
        directory, name = posixpath.split(path)
        container.exec_run(["mkdir", "-p", directory or "/"])
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(payload), 0o644, int(time.time())
            info.uid, info.gid = os.getuid(), os.getgid()
            archive.addfile(info, io.BytesIO(payload))
        if not container.put_archive(directory or "/", buffer.getvalue()):
            raise OSError(f"could not write {path} in the sandbox")
        self._record("write", path=path, bytes=len(payload))

    def note(self, text: str) -> None:
        self._record("note", content=text)

    def close(self) -> None:
        with self._lock:
            if self.container is None:
                return
            try:
                self.container.reload()
                self.oom_killed = bool(self.container.attrs.get("State", {}).get("OOMKilled"))
                self.container.remove(force=True)
            except NotFound:
                pass
            self.container = None


def run_python_agent(
    config: dict[str, Any], spec: RunSpec, agent_class: type, settings: dict[str, Any], key: str, workspace: Path, output: Path, timeout: int
) -> dict[str, Any]:
    from ray import cloudpickle

    started = time.monotonic()
    folder = output.parent
    report_path = folder / "agent_exit.json"
    report_path.unlink(missing_ok=True)
    job = {
        "agent_class": agent_class, "settings": settings, "config": config, "spec": asdict(spec), "key": key,
        "workspace": str(workspace), "output": str(output), "timeout": timeout, "deadline": time.time() + timeout,
    }
    environment = {**os.environ, EXECUTION_MARKER: spec.execution_id}
    timed_out = False
    with (folder / "agent.log").open("wb") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "cybergym_eval.agent_process"],
            stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True,
        )
        try:
            process.stdin.write(cloudpickle.dumps(job))
            process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            process.wait(timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                block_key(config, key)
            except (OSError, urllib.error.URLError):
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    remove_sandboxes(spec.execution_id)
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    error = report.get("error")
    if not timed_out and not report_path.exists():
        error = f"the agent process exited with code {process.returncode} before reporting; see agent.log"
    return {
        "seconds": round(time.monotonic() - started, 1),
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "oom_killed": bool(report.get("oom_killed")),
        "commands": report.get("commands"),
        "error": error.strip().splitlines()[-1] if error else None,
    }


def describe_agent(agent_class: type, settings: dict[str, Any], fingerprint: str) -> dict[str, Any]:
    return {
        "agent_class": f"{agent_class.__module__}.{agent_class.__qualname__}",
        "agent_fingerprint": fingerprint,
        "agent_settings": json.loads(json.dumps(settings, default=str)),
    }
