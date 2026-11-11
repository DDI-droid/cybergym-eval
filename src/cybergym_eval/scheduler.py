from __future__ import annotations

import fcntl
import logging
import os
import signal
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import docker
from docker.errors import NotFound

from . import db
from .config import RunSpec, db_path, work_id

WORK_MARKER = "CYBERGYM_EVAL_WORK"
EXECUTION_MARKER = "CYBERGYM_EVAL_EXECUTION"


class WorkFolderBusy(RuntimeError):
    pass


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@contextmanager
def work_lock(config: dict[str, Any]) -> Iterator[None]:
    lock_path = Path(config["work_root"]) / "run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise WorkFolderBusy(f"Another run is already active in {config['work_root']}.") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def ray_dir(config: dict[str, Any]) -> str:
    return f"/tmp/ray-{os.getuid()}-{work_id(config)}"


def kill_marked(marker: str) -> int:
    wanted = marker.encode()
    killed = 0
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid() or wanted not in (proc / "environ").read_bytes().split(b"\0"):
                continue
            os.kill(int(proc.name), signal.SIGKILL)
            killed += 1
        except (OSError, ProcessLookupError):
            continue
    return killed


def clean_leftovers(config: dict[str, Any], client: docker.DockerClient) -> None:
    killed = kill_marked(f"{WORK_MARKER}={work_id(config)}")
    leftovers = client.containers.list(all=True, filters={"label": f"cybergym-eval.work={work_id(config)}"})
    for container in leftovers:
        try:
            container.remove(force=True)
        except NotFound:
            pass
    if killed or leftovers:
        print(f"Cleaned up after an earlier run in this work folder: {killed} process(es), {len(leftovers)} container(s).")


def _enforce_deadline(con: Any, client: docker.DockerClient, refs_by_id: dict[str, Any], deadline: float, cancelled: set[str]) -> None:
    import ray

    now = datetime.now(timezone.utc)
    for row in db.rows(con):
        if row["status"] != "running" or row["execution_id"] not in refs_by_id:
            continue
        started = parse_time(row["started_at"])
        if started is None or (now - started).total_seconds() <= deadline:
            continue
        ray.cancel(refs_by_id[row["execution_id"]], force=True)
        kill_marked(f"{EXECUTION_MARKER}={row['execution_id']}")
        for container in client.containers.list(all=True, filters={"label": f"execution={row['execution_id']}"}):
            try:
                container.remove(force=True)
            except NotFound:
                pass
        db.mark_lost(con, row["execution_id"], f"deadline of {deadline:.0f} s exceeded", status="interrupted")
        cancelled.add(row["execution_id"])


def run_specs(
    config: dict[str, Any],
    specs: list[RunSpec],
    function: Callable[..., dict[str, Any]],
    args: tuple[Any, ...] = (),
    on_result: Callable[[RunSpec, dict[str, Any]], None] | None = None,
) -> dict[str, dict[str, Any]]:
    import ray

    results: dict[str, dict[str, Any]] = {}
    if not specs:
        return results
    con = db.connect(db_path(config))
    os.environ[WORK_MARKER] = work_id(config)
    ray.init(
        address="local",
        num_cpus=os.cpu_count() or 1,
        resources={"benchmark_slot": config["benchmark_slots"]},
        _temp_dir=ray_dir(config),
        include_dashboard=False,
        logging_level=logging.ERROR,
        log_to_driver=False,
    )
    try:
        remote = ray.remote(max_retries=0)(function)
        options: dict[str, Any] = {"num_cpus": config["task_cpus"], "resources": {"benchmark_slot": 1}}
        if config.get("scheduling_strategy"):
            options["scheduling_strategy"] = config["scheduling_strategy"]
        client = docker.from_env()
        deadline = float(config["task_deadline_seconds"])
        refs = {remote.options(**options).remote(config, asdict(spec), *args): spec for spec in specs}
        refs_by_id = {spec.execution_id: ref for ref, spec in refs.items()}
        cancelled: set[str] = set()
        pending = list(refs)
        while pending:
            ready, pending = ray.wait(pending, num_returns=1, timeout=15)
            _enforce_deadline(con, client, refs_by_id, deadline, cancelled)
            if not ready:
                continue
            spec = refs[ready[0]]
            try:
                result = ray.get(ready[0])
            except Exception as error:
                if spec.execution_id in cancelled:
                    result = {"status": "interrupted", "error": f"deadline of {deadline:.0f} s exceeded"}
                else:
                    message = f"{type(error).__name__}: {error}"
                    db.mark_lost(con, spec.execution_id, message)
                    result = {"status": "infra_error", "error": message}
            results[spec.execution_id] = result
            if on_result:
                on_result(spec, result)
        return results
    finally:
        ray.shutdown()
        os.environ.pop(WORK_MARKER, None)
        con.close()
