from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DIFFICULTIES = {"level0", "level1", "level2", "level3"}
TASK_PREFIXES = ("arvo:", "oss-fuzz:")
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
RESERVED_LABELS = {"state", "server", "taskset.json"}
REQUIRED = {"benchmark_root", "work_root", "taskset", "tasks", "difficulty", "harnesses", "models"}
MODEL_FIELDS = {"base_url", "api_key_env", "price_in_per_m", "price_out_per_m"}
# Settings that can change between runs of a label without making it a different experiment: where each model is
# served and what it costs (the harness's harness.toml names its models, and that is frozen in its code hash),
# concurrency, start spacing, ports, the credentials file and the note. They are left out of the frozen receipt
# and of the config hash, so changing them needs no new label and keeps the execution ids.
UNFROZEN_KEYS = {"models", "benchmark_slots", "start_stagger_seconds", "shim_port", "server_port", "env_file", "experiment_note"}
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


def normalize_splits(splits: Any) -> dict[str, float]:
    """Check a sample's named parts: {"train": 0.75, "test": 0.25}. Fractions above 0 that add up to exactly 1."""
    if not isinstance(splits, dict) or not splits:
        raise ValueError('tasks.splits must name at least one part, e.g. {"train": 0.75, "test": 0.25}')
    total = Fraction(0)
    for name, share in splits.items():
        _check_name("a split name", name)
        if name == "all":
            raise ValueError('"all" is reserved (split "all" runs every part); give the part another name')
        if type(share) not in (int, float) or not 0 < share <= 1:
            raise ValueError(f"tasks.splits[{name!r}] must be a fraction above 0 and at most 1, got {share!r}")
        total += Fraction(str(share))  # exact decimal arithmetic: 0.7 + 0.1 + 0.2 is exactly 1
    if total != 1:
        raise ValueError(f"tasks.splits must add up to 1; {dict(splits)} adds up to {float(total):g}")
    return dict(splits)


def split_counts(total: int, splits: dict[str, float]) -> dict[str, int]:
    """How many of `total` drawn tasks each part gets, cut in the order the parts are written.

    Each boundary is floor(total x running fraction) and the last part takes the rest, so the counts always add
    up to total.
    """
    counts: dict[str, int] = {}
    start, running = 0, Fraction(0)
    for name, share in splits.items():
        running += Fraction(str(share))
        end = total if running == 1 else math.floor(total * running)
        counts[name] = end - start
        start = end
    return counts


def sample_parts(sample: dict[str, Any]) -> dict[str, float]:
    """The named parts of a sample; none unless the config gives "splits", so a plain sample is one task set."""
    return sample.get("splits", {})


def normalize_tasks(tasks: Any, default_difficulty: str) -> list[dict[str, str]] | dict[str, Any]:
    if isinstance(tasks, dict):
        unknown = set(tasks) - {"sample", "seed", "splits"}
        if unknown:
            raise ValueError(f"Unknown keys in the tasks sample: {sorted(unknown)}")
        if type(tasks.get("sample")) is not int or tasks["sample"] < 1:
            raise ValueError("tasks.sample must be a whole number of tasks to draw")
        seed = tasks.get("seed", 0)
        if type(seed) is not int:
            raise ValueError("tasks.seed must be an integer")
        normalized: dict[str, Any] = {"sample": tasks["sample"], "seed": seed}
        if "splits" in tasks:  # only when given, so existing sampled task sets keep the same identity
            normalized["splits"] = normalize_splits(tasks["splits"])
            empty = [name for name, count in split_counts(tasks["sample"], normalized["splits"]).items() if count == 0]
            if empty:
                raise ValueError(f"tasks.sample {tasks['sample']} is too small: part(s) {empty} would get no tasks")
        return normalized
    if not isinstance(tasks, list):
        raise ValueError('tasks must be a list of task IDs or {"sample": N, "seed": S, "splits": {...}}')
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


def normalize_models(raw: Any) -> dict[str, dict[str, Any]]:
    """The run config's models table: model name -> where it is served, which credential, and its prices.

    The names are what harnesses list in their harness.toml; the rest stays on the host, with the shim.
    """
    if not isinstance(raw, dict) or not raw:
        raise ValueError('models must name at least one model, e.g. {"deepseek.v3.2": {"base_url": "https://…/v1", '
                         '"api_key_env": "OPENAI_API_KEY", "price_in_per_m": 0.62, "price_out_per_m": 1.85}}')
    table = {}
    for name, entry in raw.items():
        if not isinstance(name, str) or not name or any(c.isspace() for c in name):
            raise ValueError(f"A model name must be a non-empty string without spaces, got {name!r}")
        if not isinstance(entry, dict):
            raise ValueError(f"models[{name!r}] must be an object")
        unknown = set(entry) - MODEL_FIELDS
        if unknown:
            raise ValueError(f"Unknown keys in models[{name!r}]: {sorted(unknown)} (use {sorted(MODEL_FIELDS)})")
        url = entry.get("base_url")
        if not isinstance(url, str) or urlparse(url).scheme not in ("http", "https") or not urlparse(url).hostname:
            raise ValueError(f"models[{name!r}].base_url must be an http(s) URL, got {url!r}")
        key_env = entry.get("api_key_env")
        names = [key_env] if isinstance(key_env, str) else key_env
        if not isinstance(names, list) or not names or not all(isinstance(n, str) and n for n in names):
            raise ValueError(f"models[{name!r}].api_key_env must name the environment variable holding its credential "
                             "(or a list of names, the first one set is used)")
        for price in ("price_in_per_m", "price_out_per_m"):
            value = entry.get(price)
            if value is not None and (type(value) not in (int, float) or value < 0):
                raise ValueError(f"models[{name!r}].{price} must be a number of dollars per million tokens")
        table[name] = {"base_url": url, "api_key_env": names, "price_in_per_m": entry.get("price_in_per_m"),
                       "price_out_per_m": entry.get("price_out_per_m")}
    return table


def models_table_sha256(models: dict[str, dict[str, Any]]) -> str:
    """Identifies a models table without its secrets: names, URLs, credential variable names and prices."""
    return hashlib.sha256(json.dumps(models, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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


LISTED_HAS_NO_PARTS = (
    "A listed task set has no parts; use split=all. To divide tasks, give each part its own listed task set, "
    'or draw a sample with named parts: {"sample": N, "seed": S, "splits": {"train": 0.75, "test": 0.25}}'
)


def _check_split_name(data: dict[str, Any], parts: Any) -> None:
    if data["split"] == "all" or data["split"] in parts:
        return
    if not parts:
        raise ValueError(
            f"Task set {data['taskset']!r} is one set with no parts; use split=all, "
            'or name parts in the sample: "splits": {"train": 0.75, "test": 0.25}'
        )
    raise ValueError(f"Task set {data['taskset']!r} has parts {list(parts)}; split must be one of them or 'all', got {data['split']!r}")


def _apply_taskset(data: dict[str, Any], raw_tasks: Any) -> None:
    path = taskset_file(data)
    if not path.exists():
        if isinstance(data["tasks"], list) and data["split"] != "all":
            raise ValueError(LISTED_HAS_NO_PARTS)
        if isinstance(data["tasks"], dict):
            _check_split_name(data, sample_parts(data["tasks"]))
        return
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved["kind"] == "sample":
        if data["tasks"] != saved["sampled_from"]:
            raise ValueError(
                f"Task set {data['taskset']!r} was drawn as {saved['sampled_from']}; "
                f"this config asks for {raw_tasks!r}. Use a new taskset name."
            )
        parts = saved.get("parts", {})
        if "train" in saved and "test" in saved:  # written by the old code, which always cut a sample in half
            parts = {"train": saved["train"], "test": saved["test"]}
        _check_split_name(data, parts)
        entries = saved["tasks"] if data["split"] == "all" else parts[data["split"]]
        data["sampled_from"] = saved["sampled_from"]
    else:
        if raw_tasks != saved["tasks"]:
            raise ValueError(f"Task set {data['taskset']!r} already lists other tasks ({path}). Use a new taskset name.")
        if data["split"] != "all":
            raise ValueError(LISTED_HAS_NO_PARTS)
        entries = saved["tasks"]
    if data.get("dataset_revision") and data["dataset_revision"] != saved["dataset_revision"]:
        raise ValueError(
            f"Task set {data['taskset']!r} uses dataset revision {saved['dataset_revision']}, "
            f"but the config pins {data['dataset_revision']}"
        )
    data["dataset_revision"] = saved["dataset_revision"]
    data["tasks"] = normalize_tasks(entries, data["difficulty"])


def harness_receipt(config: dict[str, Any], harness: dict[str, Any]) -> dict[str, Any]:
    raw = {key: value for key, value in _raw_config(config).items() if key not in UNFROZEN_KEYS}
    return {**raw, "harnesses": [harness]}


def prepared_stamp(config: dict[str, Any], harness: dict[str, Any]) -> Path:
    """Written by prepare as its last step; run starts only if it matches the config it is about to run."""
    return harness_dir(config, harness) / f"prepared.{config['split']}.json"


def check_receipts(config: dict[str, Any]) -> list[tuple[Path, dict[str, Any]]]:
    """Refuse if a label was frozen with a different config file, or holds a split whose tasks overlap. Writes nothing."""
    split = config["split"]
    plan = []
    for harness in config["harnesses"]:
        folder = harness_dir(config, harness)
        target = folder / f"config.{split}.json"
        receipt = harness_receipt(config, harness)
        # Parts of one sample are disjoint, so a label may hold several; "all" overlaps every part.
        frozen = {path.name[len("config."):-len(".json")] for path in folder.glob("config.*.json")}
        clash = sorted(frozen - {"all"}) if split == "all" else sorted(frozen & {"all"})
        if clash:
            raise ValueError(f"{folder} already holds a run of split {clash[0]!r}, whose tasks overlap split {split!r}. Use a new label.")
        if target.exists() and json.loads(target.read_text(encoding="utf-8")) != receipt:
            raise ValueError(f"{target} was frozen with different settings. Use a new label for harness {harness['name']!r}.")
        plan.append((target, receipt))
    return plan


def freeze_config(config: dict[str, Any]) -> list[Path]:
    plan = check_receipts(config)
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
    _check_name("split", data["split"])  # "all", or the name of one part of a sampled task set
    raw_tasks = data["tasks"]
    data["tasks"] = normalize_tasks(raw_tasks, data["difficulty"])
    data["models"] = normalize_models(data["models"])
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
    # Domains the firewall proxy lets agents reach. None by default: agents reach models only through the shim, which
    # runs on the host and talks to the model endpoints itself, so no container needs a model host. An explicit list
    # is frozen with the label, since it decides what an agent can reach.
    data.setdefault("firewall_domains", [])
    return data


def config_hash(config: dict[str, Any], harness: dict[str, Any] | None = None) -> str:
    material = {k: v for k, v in config.items() if k not in ("config_path", "harnesses", *UNFROZEN_KEYS)}
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