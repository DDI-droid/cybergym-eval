from __future__ import annotations

import json
import random
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import docker
from docker.errors import ImageNotFound, NotFound
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from .changes import record_change
from .config import freeze_config, load_config, taskset_file
from .harnesses import snapshot_code

DATASET = "sunblaze-ucb/cybergym"
BINARY_DATASET = "sunblaze-ucb/cybergym-server-binary"
BINARY_ARCHIVE = "cybergym-server-data.7z"
DEFAULT_RUNNER = "cybergym/oss-fuzz-base-runner:latest"
PROXY_IMAGE = "ubuntu/squid:latest"
PROXY_CONTAINER = "cybergym-proxy"
INTERNAL_NETWORK = "cybergym-internal"
MIN_FREE_GB = 5
MODES = ("vul", "fix")


def dataset_revision(config: dict[str, Any]) -> str:
    return config.get("dataset_revision") or HfApi().dataset_info(DATASET).sha


def load_pool(data_dir: Path, revision: str) -> dict[str, dict[str, Any]]:
    path = hf_hub_download(DATASET, "tasks.json", repo_type="dataset", revision=revision, local_dir=data_dir)
    return {entry["task_id"]: entry for entry in json.loads(Path(path).read_text(encoding="utf-8"))}


def ensure_taskset(
    config: dict[str, Any], pool: dict[str, dict[str, Any]], revision: str, raw_tasks: Any = None, drawn_from: dict[str, Any] | None = None
) -> bool:
    path = taskset_file(config)
    if path.exists():
        return False
    if raw_tasks is None:
        raw_tasks = json.loads(Path(config["config_path"]).read_text(encoding="utf-8"))["tasks"]
    saved: dict[str, Any] = {
        "name": config["taskset"],
        "dataset_revision": revision,
        "pool_size": len(pool),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if isinstance(config["tasks"], dict):
        sample = config["tasks"]
        if sample["sample"] > len(pool):
            raise ValueError(f"Cannot sample {sample['sample']} tasks from {len(pool)}")
        chosen = random.Random(sample["seed"]).sample(sorted(pool), sample["sample"])
        half = len(chosen) // 2
        saved.update(kind="sample", sampled_from=sample, tasks=chosen, train=chosen[:half], test=chosen[half:])
        summary = f"drew {len(chosen)} of {len(pool)} tasks with seed {sample['seed']} ({half} train, {len(chosen) - half} test)"
    else:
        saved.update(kind="list", tasks=raw_tasks)
        summary = f"listed {len(config['tasks'])} task(s)"
    if drawn_from:
        saved["drawn_from"] = drawn_from
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
    print(f"Task set {config['taskset']!r}: {summary}; saved to {path}")
    return True


def check_in_pool(items: list[dict[str, str]], pool: dict[str, dict[str, Any]]) -> None:
    missing = [item["task"] for item in items if item["task"] not in pool]
    if missing:
        raise ValueError(f"Not in the CyberGym task list: {missing}")


def download_data(config: dict[str, Any], items: list[dict[str, str]], pool: dict[str, dict[str, Any]], revision: str) -> None:
    patterns = sorted({path for item in items for path in pool[item["task"]]["task_difficulty"][item["difficulty"]]})
    snapshot_download(
        repo_id=DATASET,
        repo_type="dataset",
        revision=revision,
        local_dir=config["data_dir"],
        allow_patterns=patterns,
    )
    record_change("data_dirs", config["data_dir"])


def free_gb(path: Path) -> float:
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 1e9


def free_disk_gb(client: docker.DockerClient) -> float:
    return free_gb(Path(client.info()["DockerRootDir"]).parent)


def task_binary_dir(binary_dir: Path, task: str) -> Path:
    source, task_id = task.split(":", 1)
    return binary_dir / source / task_id


def task_has_binaries(binary_dir: Path, task: str) -> bool:
    folder = task_binary_dir(binary_dir, task)
    marker = "arvo" if task.startswith("arvo:") else "metadata.json"
    return all((folder / mode / marker).is_file() and any((folder / mode / "out").glob("*")) for mode in MODES)


def runner_images(binary_dir: Path, tasks: list[str]) -> list[str]:
    images = set()
    for task in tasks:
        for mode in MODES:
            runner = task_binary_dir(binary_dir, task) / mode / "runner"
            images.add(runner.read_text(encoding="utf-8").strip() if task.startswith("arvo:") and runner.is_file() else DEFAULT_RUNNER)
    return sorted(images)


def _ensure_archive(archive: Path) -> str:
    info = HfApi().dataset_info(BINARY_DATASET, files_metadata=True)
    expected = next(item.size for item in info.siblings if item.rfilename == BINARY_ARCHIVE)
    if archive.is_file() and archive.stat().st_size == expected:
        return info.sha
    archive.unlink(missing_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    available = free_gb(archive.parent)
    if available - expected / 1e9 < MIN_FREE_GB:
        raise RuntimeError(f"{available:.1f} GB free; the binary archive needs {expected / 1e9:.1f} GB plus {MIN_FREE_GB} GB in reserve")
    print(f"Downloading {BINARY_ARCHIVE} ({expected / 1e9:.1f} GB) from {BINARY_DATASET}...")
    hf_hub_download(BINARY_DATASET, BINARY_ARCHIVE, repo_type="dataset", revision=info.sha, local_dir=archive.parent)
    return info.sha


def _archive_entries(archive: Path) -> list[tuple[str, int]]:
    listing = subprocess.run(["7z", "l", "-ba", "-slt", str(archive)], check=True, capture_output=True, text=True).stdout
    entries, path = [], None
    for line in listing.splitlines():
        if line.startswith("Path = "):
            path = line[7:]
        elif line.startswith("Size = ") and path is not None:
            entries.append((path, int(line[7:] or 0)))
            path = None
    return entries


def _locate(entries: list[tuple[str, int]], wanted: set[str]) -> tuple[dict[str, str], dict[str, int]]:
    roots: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for path, size in entries:
        parts = path.split("/")
        for index, part in enumerate(parts[:-1]):
            if part in ("arvo", "oss-fuzz"):
                task = f"{part}:{parts[index + 1]}"
                if task in wanted:
                    roots[task] = "/".join(parts[: index + 2])
                    sizes[task] = sizes.get(task, 0) + size
                break
    return roots, sizes


def ensure_binaries(config: dict[str, Any], tasks: list[str], keep_archive: bool = False) -> None:
    binary_dir = Path(config["binary_dir"])
    missing = [task for task in tasks if not task_has_binaries(binary_dir, task)]
    if not missing:
        print(f"Binaries present for all {len(tasks)} task(s) in {binary_dir}")
        return
    archive = binary_dir.parent / BINARY_ARCHIVE
    revision = _ensure_archive(archive)
    record_change("data_dirs", str(binary_dir.parent))
    roots, sizes = _locate(_archive_entries(archive), set(missing))
    absent = [task for task in missing if task not in roots]
    if absent:
        raise RuntimeError(f"Not in the binary archive: {absent}")
    needed = sum(sizes.values()) / 1e9
    available = free_gb(binary_dir.parent)
    if available - needed < MIN_FREE_GB:
        raise RuntimeError(f"{available:.1f} GB free; extracting {len(missing)} task(s) needs {needed:.1f} GB plus {MIN_FREE_GB} GB in reserve")
    print(f"Extracting {len(missing)} task(s), {needed:.2f} GB, from {archive.name}...")
    staging = binary_dir.parent / ".extracting"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    include = binary_dir.parent / ".include.txt"
    include.write_text("\n".join(sorted(roots.values())) + "\n", encoding="utf-8")
    subprocess.run(["7z", "x", "-y", "-bd", f"-o{staging}", str(archive), f"@{include}"], check=True, stdout=subprocess.DEVNULL)
    for task, root in roots.items():
        target = task_binary_dir(binary_dir, task)
        shutil.rmtree(target, ignore_errors=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging / root), str(target))
    shutil.rmtree(staging, ignore_errors=True)
    include.unlink(missing_ok=True)
    source = binary_dir / "source.json"
    known = json.loads(source.read_text(encoding="utf-8")) if source.exists() else {"tasks": {}}
    known.update(dataset=BINARY_DATASET, archive=BINARY_ARCHIVE, revision=revision)
    known["tasks"].update({task: sizes[task] for task in roots})
    source.write_text(json.dumps(known, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    incomplete = [task for task in missing if not task_has_binaries(binary_dir, task)]
    if incomplete:
        raise RuntimeError(f"Extraction left incomplete binaries for: {incomplete}")
    if keep_archive:
        print(f"Keeping {archive} ({archive.stat().st_size / 1e9:.1f} GB); delete it once every task set is prepared.")
    else:
        archive.unlink()
        print(f"Deleted {archive.name}; prepare downloads it again if a new task needs binaries.")


def pull_images(client: docker.DockerClient, images: list[str], quiet: bool = False) -> None:
    for image in images:
        try:
            client.images.get(image)
            if not quiet:
                print(f"Have {image}")
            continue
        except ImageNotFound:
            pass
        available = free_disk_gb(client)
        if available < MIN_FREE_GB:
            raise RuntimeError(f"Only {available:.1f} GB free; not pulling {image}")
        if not quiet:
            print(f"Pulling {image} ({available:.0f} GB free)...")
        repo, tag = image.rsplit(":", 1)
        client.images.pull(repo, tag=tag)
        record_change("images", image)


def start_firewall(config: dict[str, Any], client: docker.DockerClient) -> None:
    try:
        client.networks.get(INTERNAL_NETWORK)
    except NotFound:
        record_change("firewall_started")
    pull_images(client, [PROXY_IMAGE])
    cmd = [str(Path(config["benchmark_root"]) / ".venv" / "bin" / "python"), "-m", "cybergym.firewall", "update"]
    for domain in config["firewall_domains"]:
        cmd.extend(["--domain", domain])
    subprocess.run(cmd, check=True)


def stage_tasks(
    config: dict[str, Any], client: docker.DockerClient, pool: dict[str, dict[str, Any]], revision: str,
    skip_binaries: bool = False, keep_archive: bool = False,
) -> None:
    items = config["tasks"]
    print(f"Downloading data for {len(items)} task(s) at dataset revision {revision[:12]}...")
    download_data(config, items, pool, revision)
    if skip_binaries:
        print("Skipping task binaries and runner images (--skip-binaries).")
        return
    tasks = [item["task"] for item in items]
    ensure_binaries(config, tasks, keep_archive)
    pull_images(client, runner_images(Path(config["binary_dir"]), tasks))


def prepare(config: dict[str, Any], skip_firewall: bool = False, skip_binaries: bool = False, keep_archive: bool = False) -> int:
    data_dir = Path(config["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    revision = dataset_revision(config)
    pool = load_pool(data_dir, revision)
    if ensure_taskset(config, pool, revision):
        config = load_config(config["config_path"])
    items = config["tasks"]
    check_in_pool(items, pool)
    receipts = freeze_config(config)
    print(f"Split {config['split']}: {len(items)} task(s) x {len(config['harnesses'])} harness(es)")
    for receipt in receipts:
        print(f"  receipt {receipt}")
    for harness in config["harnesses"]:
        print(f"  code of {harness['label']} frozen: {snapshot_code(config, harness)[:12]}")

    client = docker.from_env()
    if config["use_firewall"] and skip_firewall:
        print("Skipping firewall setup (--skip-firewall); reusing the existing proxy.")
    elif config["use_firewall"]:
        print(f"Starting the CyberGym firewall (allowing {', '.join(config['firewall_domains']) or 'only its defaults'})...")
        start_firewall(config, client)

    stage_tasks(config, client, pool, revision, skip_binaries, keep_archive)
    print("Subset preparation complete.")
    return 0
