from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DIFFICULTIES = {"level0", "level1", "level2", "level3"}
PROVIDERS = {"anthropic", "bedrock", "openai-compatible"}
SPLITS = {"train", "test", "all"}
TASK_PREFIXES = ("arvo:", "oss-fuzz:")
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
RESERVED_LABELS = {"state", "server", "taskset.json"}
REQUIRED = {"benchmark_root", "work_root", "taskset", "tasks", "difficulty", "harnesses", "model_provider"}
DEFAULT_AGENT_TIMEOUT_SECONDS = 3600
DEADLINE_MARGIN_SECONDS = 900


def _load_env_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


@dataclass(frozen=True)
class RunSpec:
    task: str
    difficulty: str
    label: str
    execution_id: str
    folder: str = ""


def _check_name(key: str, value: Any) -> None:
    if not isinstance(value, str) or not NAME_PATTERN.match(value):
        raise ValueError(f"{key} must be letters, digits, '.', '_' or '-', got: {value!r}")


def _check_task_id(task: Any) -> None:
    if not isinstance(task, str) or not task.startswith(TASK_PREFIXES):
        raise ValueError(f"Task IDs must look like arvo:1065 or oss-fuzz:42535468, got: {task!r}")


def _check_difficulty(difficulty: Any) -> None:
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"Unsupported difficulty: {difficulty!r} (use one of {sorted(DIFFICULTIES)})")


def normalize_tasks(tasks: Any, default_difficulty: str) -> list[dict[str, str]] | dict[str, int]:
    if isinstance(tasks, dict):
        unknown = set(tasks) - {"sample", "seed"}
        if unknown:
            raise ValueError(f"Unknown keys in the tasks sample: {sorted(unknown)}")
        if type(tasks.get("sample")) is not int or tasks["sample"] < 2:
            raise ValueError("tasks.sample must be an integer of at least 2 (it is split into train and test)")
        seed = tasks.get("seed", 0)
        if type(seed) is not int:
            raise ValueError("tasks.seed must be an integer")
        return {"sample": tasks["sample"], "seed": seed}
    if not isinstance(tasks, list):
        raise ValueError('tasks must be a list of task IDs or {"sample": N, "seed": S}')
    if not tasks:
        raise ValueError("tasks must contain at least one CyberGym task")
    items = []
    for entry in tasks:
        if isinstance(entry, dict):
            unknown = set(entry) - {"task", "difficulty"}
            if unknown:
                raise ValueError(f"Unknown keys in a task entry: {sorted(unknown)}")
            task, difficulty = entry.get("task"), entry.get("difficulty", default_difficulty)
        else:
            task, difficulty = entry, default_difficulty
        _check_task_id(task)
        _check_difficulty(difficulty)
        items.append({"task": task, "difficulty": difficulty})
    names = [item["task"] for item in items]
    if len(set(names)) != len(names):
        raise ValueError("A task may appear only once in a config; run another level under a new label")
    return items


def normalize_harnesses(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("harnesses must be a non-empty list of harness names or objects")
    harnesses = []
    for entry in raw:
        if isinstance(entry, str):
            entry = {"name": entry}
        if not isinstance(entry, dict):
            raise ValueError(f"Each harness must be a name or an object, got: {entry!r}")
        unknown = set(entry) - {"name", "label", "settings"}
        if unknown:
            raise ValueError(f"Unknown keys in a harness entry: {sorted(unknown)} (time limits are set for the whole eval)")
        name = entry.get("name")
        _check_name("harness name", name)
        label = entry.get("label", name)
        _check_name("harness label", label)
        if label in RESERVED_LABELS:
            raise ValueError(f"{label!r} is reserved; choose another label")
        settings = entry.get("settings", {})
        if not isinstance(settings, dict):
            raise ValueError(f"settings of harness {label!r} must be a JSON object")
        harnesses.append({"name": name, "label": label, "settings": settings})
    labels = [harness["label"] for harness in harnesses]
    if len(set(labels)) != len(labels):
        raise ValueError("Two harnesses share a label; give one of them a different label")
    return harnesses


def harness_by_label(config: dict[str, Any], label: str) -> dict[str, Any]:
    for harness in config["harnesses"]:
        if harness["label"] == label:
            return harness
    raise KeyError(f"No harness labelled {label!r} in this config")


def task_folder_name(task: str) -> str:
    return task.replace(":", "_")


def work_id(config: dict[str, Any]) -> str:
    return hashlib.sha256(config["work_root"].encode()).hexdigest()[:10]


def taskset_dir(config: dict[str, Any]) -> Path:
    return Path(config["work_root"]) / config["taskset"]


def taskset_file(config: dict[str, Any]) -> Path:
    return taskset_dir(config) / "taskset.json"


def harness_dir(config: dict[str, Any], harness: dict[str, Any]) -> Path:
    return taskset_dir(config) / harness["label"]


def execution_dir(config: dict[str, Any], harness: dict[str, Any], task: str) -> Path:
    return harness_dir(config, harness) / task_folder_name(task)


def db_path(config: dict[str, Any]) -> Path:
    return taskset_dir(config) / "state" / "runs.sqlite3"


def harness_definition_dir(config: dict[str, Any], harness: dict[str, Any]) -> Path:
    return Path(config["harness_repo"]) / "harnesses" / harness["name"]


def _raw_config(config: dict[str, Any]) -> dict[str, Any]:
    return json.loads(Path(config["config_path"]).read_text(encoding="utf-8"))


def _apply_taskset(data: dict[str, Any], raw_tasks: Any) -> None:
    path = taskset_file(data)
    if not path.exists():
        if isinstance(data["tasks"], list) and data["split"] != "all":
            raise ValueError("A listed task set has no train/test halves; use split=all")
        return
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved["kind"] == "sample":
        if data["tasks"] != saved["sampled_from"]:
            raise ValueError(
                f"Task set {data['taskset']!r} was drawn as {saved['sampled_from']}; "
                f"this config asks for {raw_tasks!r}. Use a new taskset name."
            )
        entries = saved["tasks"] if data["split"] == "all" else saved[data["split"]]
        data["sampled_from"] = saved["sampled_from"]
    else:
        if raw_tasks != saved["tasks"]:
            raise ValueError(f"Task set {data['taskset']!r} already lists other tasks ({path}). Use a new taskset name.")
        if data["split"] != "all":
            raise ValueError("A listed task set has no train/test halves; use split=all")
        entries = saved["tasks"]
    if data.get("dataset_revision") and data["dataset_revision"] != saved["dataset_revision"]:
        raise ValueError(
            f"Task set {data['taskset']!r} uses dataset revision {saved['dataset_revision']}, "
            f"but the config pins {data['dataset_revision']}"
        )
    data["dataset_revision"] = saved["dataset_revision"]
    data["tasks"] = normalize_tasks(entries, data["difficulty"])


def harness_receipt(config: dict[str, Any], harness: dict[str, Any]) -> dict[str, Any]:
    return {**_raw_config(config), "harnesses": [harness]}


def freeze_config(config: dict[str, Any]) -> list[Path]:
    split = config["split"]
    overlapping = ["train", "test"] if split == "all" else ["all"]
    plan = []
    for harness in config["harnesses"]:
        folder = harness_dir(config, harness)
        target = folder / f"config.{split}.json"
        receipt = harness_receipt(config, harness)
        clash = [name for name in overlapping if (folder / f"config.{name}.json").exists()]
        if clash:
            raise ValueError(f"{folder} already holds a {clash[0]} run whose tasks overlap split {split!r}. Use a new label.")
        if target.exists() and json.loads(target.read_text(encoding="utf-8")) != receipt:
            raise ValueError(f"{target} was frozen with different settings. Use a new label for harness {harness['name']!r}.")
        plan.append((target, receipt))
    for target, receipt in plan:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return [target for target, _ in plan]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED - data.keys())
    if missing:
        raise ValueError(f"Missing configuration keys: {', '.join(missing)}")
    data["harnesses"] = normalize_harnesses(data["harnesses"])
    data["config_path"] = str(config_path)
    return finish_config(data, config_path.parent)


def finish_config(data: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    _check_name("taskset", data["taskset"])
    _check_difficulty(data["difficulty"])
    data.setdefault("split", "all")
    if data["split"] not in SPLITS:
        raise ValueError(f"Unsupported split: {data['split']!r} (use one of {sorted(SPLITS)})")
    raw_tasks = data["tasks"]
    data["tasks"] = normalize_tasks(raw_tasks, data["difficulty"])
    if data["model_provider"] not in PROVIDERS:
        raise ValueError(f"Unsupported model_provider: {data['model_provider']}")
    data["benchmark_root"] = str(Path(data["benchmark_root"]).expanduser().resolve())
    data["work_root"] = str(Path(data["work_root"]).expanduser().resolve())
    data["data_dir"] = str(Path(data.get("data_dir", "~/cybergym_data")).expanduser().resolve())
    data["harness_repo"] = str(Path(data.get("harness_repo", "~/cybergym-harnesses")).expanduser().resolve())
    data["binary_dir"] = str(Path(data.get("binary_dir", "~/cybergym_bin/cybergym-server-data")).expanduser().resolve())
    _apply_taskset(data, raw_tasks)
    data.setdefault("agent_timeout_seconds", DEFAULT_AGENT_TIMEOUT_SECONDS)
    if data.get("task_deadline_seconds") is None:
        data["task_deadline_seconds"] = data["agent_timeout_seconds"] + DEADLINE_MARGIN_SECONDS
    for key in ("agent_timeout_seconds", "task_deadline_seconds"):
        if type(data[key]) is not int or data[key] < 1:
            raise ValueError(f"{key} must be a positive whole number of seconds")
    data.setdefault("task_cpus", 2)
    data.setdefault("task_memory_gb", 6)
    data.setdefault("benchmark_slots", 1)
    data.setdefault("use_firewall", True)
    data.setdefault("model", "")
    data.setdefault("aws_region", "us-east-1")
    data.setdefault("openai_base_url", "")
    data.setdefault("start_stagger_seconds", 0)
    data.setdefault("scheduling_strategy", "")
    data.setdefault("shim_port", 4100)
    data.setdefault("server_port", 8666)
    env_file = data.get("env_file")
    if env_file:
        env_path = Path(env_file).expanduser()
        if not env_path.is_absolute():
            env_path = base_dir / env_path
        data["env_file"] = str(env_path.resolve())
        _load_env_file(Path(data["env_file"]))
    if data["model_provider"] == "openai-compatible":
        data["openai_base_url"] = (
            data["openai_base_url"]
            or os.environ.get("OPENAI_BASE_URL")
            or f"https://bedrock-runtime.{data['aws_region']}.amazonaws.com/openai/v1"
        )
        os.environ.setdefault("OPENAI_BASE_URL", data["openai_base_url"])
    if "firewall_domains" not in data:
        default_domains = []
        if data["model_provider"] == "bedrock":
            default_domains.append(f"bedrock-runtime.{data['aws_region']}.amazonaws.com")
        elif data["model_provider"] == "openai-compatible" and data["openai_base_url"]:
            host = urlparse(data["openai_base_url"]).hostname
            if host:
                default_domains.append(host)
        data["firewall_domains"] = default_domains
    return data


def config_hash(config: dict[str, Any], harness: dict[str, Any] | None = None) -> str:
    material = {k: v for k, v in config.items() if k not in ("config_path", "harnesses")}
    material["harness"] = harness
    payload = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_specs(config: dict[str, Any]) -> list[RunSpec]:
    if isinstance(config["tasks"], dict):
        raise ValueError(f"Task set {config['taskset']!r} has not been drawn yet; run prepare first")
    digests = {harness["label"]: config_hash(config, harness) for harness in config["harnesses"]}
    specs: list[RunSpec] = []
    for item in config["tasks"]:
        for harness in config["harnesses"]:
            raw = f"{digests[harness['label']]}:{item['task']}:{item['difficulty']}".encode()
            execution_id = hashlib.sha256(raw).hexdigest()[:20]
            specs.append(RunSpec(item["task"], item["difficulty"], harness["label"], execution_id))
    return specs