from __future__ import annotations

import hashlib
import inspect
import json
import random
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import docker

from . import db
from .config import RunSpec, _check_name, db_path, finish_config, task_folder_name, taskset_dir
from .prepare import ensure_taskset, load_pool, stage_tasks
from .scheduler import clean_leftovers, run_specs, work_lock
from .server import start_server
from .shim import start_shim
from .worker import execute_python, execute_wrapped

DEFAULT_REVISION = "bde190ded494e52bc684b66073b436c9d992c7c6"
PATH_KEYS = ("benchmark_root", "data_dir", "harness_repo", "binary_dir")


@dataclass(frozen=True)
class TaskSet:
    name: str
    tasks: tuple[str, ...]
    config: dict[str, Any] = field(repr=False)

    def __iter__(self) -> Iterator[str]:
        return iter(self.tasks)

    def __len__(self) -> int:
        return len(self.tasks)


@dataclass(frozen=True)
class Rollout:
    task: str
    status: str
    passed: bool
    folder: Path
    result: dict[str, Any] = field(repr=False)

    @property
    def reason(self) -> str | None:
        return self.result.get("reason")

    @property
    def error(self) -> str | None:
        return self.result.get("error")

    @property
    def final(self) -> dict[str, Any] | None:
        return self.result.get("final")

    @property
    def usage(self) -> dict[str, Any]:
        return self.result.get("usage") or {}

    @property
    def feedback(self) -> str:
        if self.passed:
            return "Passed: the PoC crashed the vulnerable build and not the fixed build."
        parts = [self.reason or self.error or self.status]
        final = self.final or {}
        if final:
            parts.append(f"vulnerable build exit code {final.get('vul_exit_code')}, fixed build exit code {final.get('fix_exit_code')}")
        agent_error = (self.result.get("agent") or {}).get("error")
        if agent_error:
            parts.append(f"agent error: {agent_error}")
        return "; ".join(str(part) for part in parts)


def _task_ids(items: TaskSet | Iterable[str | TaskSet]) -> list[str]:
    if isinstance(items, TaskSet):
        return list(items.tasks)
    found: list[str] = []
    for item in items:
        found.extend(item.tasks if isinstance(item, TaskSet) else [item])
    return found


def _pickle_by_value(agent_class: type) -> None:
    module = inspect.getmodule(agent_class)
    if module is None or module.__name__ == "__main__":
        return
    location = getattr(module, "__file__", "") or ""
    if "site-packages" in location or "dist-packages" in location:
        return
    from ray import cloudpickle

    cloudpickle.register_pickle_by_value(module)


class Agent:
    def __init__(self, agent_class: type, name: str | None, settings: dict[str, Any]):
        self.container_harness = agent_class if getattr(agent_class, "protocol", None) == "cybergym.container.v1" else None
        if self.container_harness is not None:
            self.container_harness.container_code(settings)
        elif not inspect.isclass(agent_class) or not callable(getattr(agent_class, "solve", None)):
            raise TypeError("gym.agent() takes a solve(task, box, lm) class or a built DockerHarness")
        self.agent_class = agent_class
        self.name = name or (self.container_harness.name if self.container_harness else agent_class.__name__) 
        _check_name("agent name", self.name)
        try:
            encoded = json.dumps(settings, sort_keys=True)
        except TypeError:
            raise TypeError("agent settings must be JSON values: text, numbers, lists and dicts") from None
        self.settings = json.loads(encoded)
        if self.container_harness:
            self.source = json.dumps({"image_id": self.container_harness.image_id, "code_sha256": self.container_harness.fingerprint}, sort_keys=True)
        else:
            try:
                self.source = inspect.getsource(agent_class)
            except (OSError, TypeError):
                self.source = f"{agent_class.__module__}.{agent_class.__qualname__}"
        self.fingerprint = hashlib.sha256(f"{self.source}\n{encoded}".encode()).hexdigest()[:16]
        _pickle_by_value(type(agent_class) if self.container_harness else agent_class)

    def rollout(self, taskset: TaskSet, task: str) -> Rollout:
        return self.rollouts(taskset, [task])[0]

    def rollouts(self, taskset: TaskSet, tasks: Iterable[str] | None = None) -> list[Rollout]:
        from .cli import preflight

        config = taskset.config
        chosen = list(taskset.tasks if tasks is None else tasks)
        unknown = [task for task in chosen if task not in taskset.tasks]
        if unknown:
            raise ValueError(f"Not in task set {taskset.name!r}: {unknown}")
        start_server(config)
        start_shim(config)
        if preflight(config, quiet=True):
            raise RuntimeError("Preflight failed; see the FAIL lines above")
        base = taskset_dir(config) / self.name
        self._save_code(base)
        specs: list[RunSpec] = []
        with work_lock(config):
            clean_leftovers(config, docker.from_env())
            con = db.connect(db_path(config))
            for task in chosen:
                execution_id = uuid.uuid4().hex[:20]
                folder = base / task_folder_name(task) / execution_id
                specs.append(RunSpec(task, config["difficulty"], self.name, execution_id, str(folder)))
                db.register(con, execution_id, self.fingerprint, self.name, config["split"], task, config["difficulty"])
            con.close()
            function = execute_wrapped if self.container_harness else execute_python
            results = run_specs(config, specs, function, args=(self.agent_class, self.settings, self.fingerprint))
        return [
            Rollout(
                task=spec.task,
                status=(results.get(spec.execution_id) or {}).get("status", "infra_error"),
                passed=(results.get(spec.execution_id) or {}).get("status") == "passed",
                folder=Path(spec.folder),
                result=results.get(spec.execution_id) or {"status": "infra_error", "error": "no result came back"},
            )
            for spec in specs
        ]

    def _save_code(self, base: Path) -> None:
        target = base / "code" / f"{self.fingerprint}.json"
        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        record = {"class": (self.container_harness.name if self.container_harness else f"{self.agent_class.__module__}.{self.agent_class.__qualname__}"), "settings": self.settings, "source": self.source}
        target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


class Gym:
    def __init__(
        self,
        work_root: str = "~/cybergym-gym",
        *,
        difficulty: str = "level2",
        model: str = "deepseek.v3.2",
        env_file: str = "~/.secrets/bedrock.env",
        benchmark_slots: int = 4,
        task_cpus: float = 2,
        task_memory_gb: float = 6,
        agent_timeout_seconds: int = 3600,
        task_deadline_seconds: int | None = None,
        price_in_per_m: float = 0.62,
        price_out_per_m: float = 1.85,
        server_port: int = 8680,
        shim_port: int = 4120,
        dataset_revision: str = DEFAULT_REVISION,
        held_out: TaskSet | Iterable[str | TaskSet] = (),
        **paths: str,
    ):
        unknown = sorted(set(paths) - set(PATH_KEYS))
        if unknown:
            raise TypeError(f"Unknown Gym arguments: {unknown}")
        self.held_out = frozenset(_task_ids(held_out))
        self._base: dict[str, Any] = {
            "benchmark_root": "~/cybergym",
            "data_dir": "~/cybergym_data",
            "harness_repo": "~/cybergym-harnesses",
            "binary_dir": "~/cybergym_bin/cybergym-server-data",
            **paths,
            "work_root": work_root,
            "env_file": env_file,
            "difficulty": difficulty,
            "dataset_revision": dataset_revision,
            "model_provider": "openai-compatible",
            "model": model,
            "aws_region": "us-east-1",
            "benchmark_slots": benchmark_slots,
            "task_cpus": task_cpus,
            "task_memory_gb": task_memory_gb,
            "agent_timeout_seconds": agent_timeout_seconds,
            "price_in_per_m": price_in_per_m,
            "price_out_per_m": price_out_per_m,
            "server_port": server_port,
            "shim_port": shim_port,
            "use_firewall": True,
            "split": "all",
            "harnesses": [],
        }
        if task_deadline_seconds is not None:
            self._base["task_deadline_seconds"] = task_deadline_seconds
        self.work_root = Path(work_root).expanduser().resolve()

    def _config(self, name: str, tasks: list[str]) -> dict[str, Any]:
        return finish_config({**self._base, "taskset": name, "tasks": list(tasks)}, Path.home())

    def taskset(
        self, name: str, tasks: Iterable[str] | None = None, *, sample: int | None = None, seed: int = 0,
        exclude: TaskSet | Iterable[str | TaskSet] = (),
    ) -> TaskSet:
        _check_name("task set name", name)
        revision = self._base["dataset_revision"]
        data_dir = Path(self._base["data_dir"]).expanduser()
        data_dir.mkdir(parents=True, exist_ok=True)
        pool = load_pool(data_dir, revision)
        drawn_from = None
        if tasks is None and sample is None:
            saved_path = self.work_root / name / "taskset.json"
            if not saved_path.exists():
                raise ValueError(f"There is no task set {name!r} in {self.work_root}; create it with tasks=[...] or sample=n")
            saved = json.loads(saved_path.read_text(encoding="utf-8"))
            if saved.get("kind") != "list":
                raise ValueError(f"Task set {name!r} was drawn by a sample config; use it through the CLI")
            chosen = list(saved["tasks"])
        else:
            if tasks is None:
                raise ValueError("sample= draws from tasks=: pass the task IDs to draw from (your training IDs), never the whole pool")
            excluded = set(_task_ids(exclude))
            candidates = [task for task in tasks if task not in excluded]
            missing = [task for task in candidates if task not in pool]
            if missing:
                raise ValueError(f"Not in the CyberGym task list: {missing}")
            if sample is None:
                chosen = candidates
                drawn_from = {"excluded": len(excluded)} if excluded else None
            else:
                if type(sample) is not int or not 1 <= sample <= len(candidates):
                    raise ValueError(f"sample must be a whole number from 1 to {len(candidates)}")
                chosen = random.Random(seed).sample(sorted(candidates), sample)
                drawn_from = {
                    "sample": sample, "seed": seed, "candidates": len(candidates),
                    "excluded": len(excluded), "dataset_revision": revision,
                }
        leaked = sorted(set(chosen) & self.held_out)
        if leaked:
            raise ValueError(f"Task set {name!r} contains held-out tasks: {leaked}")
        config = self._config(name, chosen)
        if ensure_taskset(config, pool, revision, raw_tasks=chosen, drawn_from=drawn_from):
            config = self._config(name, chosen)
        stage_tasks(config, docker.from_env(), pool, revision)
        return TaskSet(name, tuple(chosen), config)

    def agent(self, agent_class: type, name: str | None = None, **settings: Any) -> Agent:
        return Agent(agent_class, name, settings)
