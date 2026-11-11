from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import docker
from docker.errors import ImageNotFound, NotFound

from . import db
from .config import (
    build_specs,
    config_hash,
    db_path,
    harness_by_label,
    harness_definition_dir,
    harness_dir,
    load_config,
)
from .failures import upstream_failure
from .harnesses import read_manifest
from .prepare import MIN_FREE_GB, PROXY_CONTAINER, free_disk_gb, prepare, runner_images, task_has_binaries
from .scheduler import WorkFolderBusy, clean_leftovers, run_specs, work_lock
from .scheduler import parse_time as _parse_time
from .server import server_status, start_server, stop_server
from .shim import shim_status, start_shim, stop_shim

Check = tuple[str, bool, str]
COMPLETED_STATUSES = {"passed", "benchmark_failed"}


def _short(items: list[str]) -> str:
    return ", ".join(items[:3]) + (f" and {len(items) - 3} more" if len(items) > 3 else "")


def _mmap_bits() -> str:
    try:
        return Path("/proc/sys/vm/mmap_rnd_bits").read_text().strip()
    except OSError:
        probe = subprocess.run(["sudo", "-n", "sysctl", "-n", "vm.mmap_rnd_bits"], capture_output=True, text=True)
        return probe.stdout.strip() if probe.returncode == 0 else "unknown"


def _has_image(client: docker.DockerClient, image: str) -> bool:
    try:
        client.images.get(image)
        return True
    except ImageNotFound:
        return False


def _task_checks(config: dict[str, Any], client: docker.DockerClient) -> list[Check]:
    if isinstance(config["tasks"], dict):
        return [("task set drawn", False, "run prepare first")]
    data_dir = Path(config["data_dir"])
    catalogue = data_dir / "tasks.json"
    if not catalogue.is_file():
        return [("task catalogue", False, f"{catalogue} missing; run prepare first")]
    pool = {entry["task_id"]: entry for entry in json.loads(catalogue.read_text(encoding="utf-8"))}

    def has_data(item: dict[str, str]) -> bool:
        entry = pool.get(item["task"])
        return entry is not None and all((data_dir / path).is_file() for path in entry["task_difficulty"][item["difficulty"]])

    missing_data = [item["task"] for item in config["tasks"] if not has_data(item)]
    binary_dir = Path(config["binary_dir"])
    tasks = [item["task"] for item in config["tasks"]]
    missing_binaries = [task for task in tasks if not task_has_binaries(binary_dir, task)]
    missing_runners = [image for image in runner_images(binary_dir, tasks) if not _has_image(client, image)]
    count = len(tasks)
    return [
        ("task catalogue", True, str(catalogue)),
        (f"task data ({count} task(s))", not missing_data, "present" if not missing_data else f"missing: {_short(missing_data)}"),
        (f"grading binaries ({count} task(s))", not missing_binaries,
         str(binary_dir) if not missing_binaries else f"missing: {_short(missing_binaries)}; run prepare"),
        ("grading runner images", not missing_runners, "present" if not missing_runners else f"missing: {_short(missing_runners)}; run prepare"),
    ]


def _harness_checks(config: dict[str, Any], client: docker.DockerClient | None) -> list[Check]:
    checks = []
    for harness in config["harnesses"]:
        definition = harness_definition_dir(config, harness)
        try:
            manifest = read_manifest(definition)
        except ValueError as error:
            checks.append((f"harness {harness['label']}", False, str(error)))
            continue
        checks.append((f"harness {harness['label']} definition", True, str(definition / "harness.toml")))
        if client:
            present = _has_image(client, manifest["image"])
            detail = manifest["image"] if present else f"{manifest['image']} missing; run ~/cybergym-harnesses/build.sh"
            checks.append((f"harness {harness['label']} image", present, detail))
    return checks


def _firewall_checks(config: dict[str, Any], client: docker.DockerClient) -> list[Check]:
    if not config["use_firewall"]:
        return [("firewall", True, "off in config: agents get the whole internet")]
    try:
        proxy = client.containers.get(PROXY_CONTAINER)
    except NotFound:
        return [("firewall proxy", False, "not running; run prepare")]
    running = proxy.status == "running"
    return [("firewall network and proxy", running, f"{PROXY_CONTAINER} {proxy.status}")]


def _service_checks(config: dict[str, Any]) -> list[Check]:
    server, shim = server_status(config), shim_status(config)
    checks = []
    for name, state, command in (("submission server", server, "server"), ("LLM shim", shim, "shim")):
        ours = bool(state["listening"] and state["pid"])
        if ours:
            detail = f"{state['url']} (pid {state['pid']})"
        elif state["listening"]:
            detail = f"something else answers on {state['url']}, not the {command} started for {config['work_root']}"
        else:
            detail = f"not running; run: cybergym-eval {command} start"
        checks.append((name, ours, detail))
    if checks[0][1] and server["binary_dir"] != config["binary_dir"]:
        checks[0] = (checks[0][0], False, f"running with --binary_dir {server['binary_dir']}, expected {config['binary_dir']}; restart it")
    return checks


def _credential_check(config: dict[str, Any]) -> Check:
    provider = config["model_provider"]
    found = {
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "openai-compatible": bool(
            os.getenv("OPENAI_BASE_URL") and (os.getenv("OPENAI_API_KEY") or os.getenv("AWS_BEARER_TOKEN_BEDROCK"))
        ),
        "bedrock": bool(os.getenv("AWS_BEARER_TOKEN_BEDROCK") or os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_PROFILE")),
    }[provider]
    return (f"{provider} credentials", found, "found" if found else "not found")


def preflight(config: dict[str, Any], quiet: bool = False) -> int:
    root = Path(config["benchmark_root"])
    checks: list[Check] = [
        ("benchmark checkout", (root / "src" / "cybergym").is_dir(), str(root)),
        ("benchmark virtualenv", (root / ".venv" / "bin" / "python").is_file(), str(root / ".venv")),
    ]
    try:
        client = docker.from_env()
        client.ping()
    except Exception:
        client = None
    checks.append(("Docker daemon", client is not None, "available" if client else "unavailable"))
    bits = _mmap_bits()
    checks.append(("ASLR vm.mmap_rnd_bits = 28", bits == "28", bits))
    memory_gib = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    checks.append(("host memory >= 8 GiB", memory_gib >= 7.5, f"{memory_gib:.1f} GiB"))
    checks.extend(_harness_checks(config, client))
    if client:
        free_gb = free_disk_gb(client)
        checks.append(("free disk", free_gb >= MIN_FREE_GB, f"{free_gb:.1f} GB free, at least {MIN_FREE_GB} GB needed"))
        checks.extend(_task_checks(config, client))
        checks.extend(_firewall_checks(config, client))
        checks.extend(_service_checks(config))
    checks.append(_credential_check(config))
    priced = config.get("price_in_per_m") is not None and config.get("price_out_per_m") is not None
    detail = (f"${config.get('price_in_per_m')} / ${config.get('price_out_per_m')} per 1M tokens in / out"
              if priced else "set price_in_per_m and price_out_per_m, or the report cannot show cost")
    checks.append(("model prices (cost report)", priced, detail))
    width = max(len(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        if not (quiet and ok):
            print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    failed = [name for name, ok, _ in checks if not ok]
    if failed:
        print(f"\nPreflight failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def check_model(config: dict[str, Any]) -> int:
    if config["model_provider"] != "openai-compatible":
        print("check-model supports the openai-compatible provider only.", file=sys.stderr)
        return 1
    key = os.getenv("OPENAI_API_KEY") or os.getenv("AWS_BEARER_TOKEN_BEDROCK") or ""
    body = {"model": config["model"], "max_tokens": 16, "messages": [{"role": "user", "content": "Reply with exactly: ok"}]}
    request = urllib.request.Request(
        f"{config['openai_base_url'].rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            reply = json.loads(response.read())
    except urllib.error.HTTPError as error:
        print(f"Model check failed: HTTP {error.code} {error.read().decode(errors='replace')[:300]}", file=sys.stderr)
        return 1
    answer = reply["choices"][0]["message"]["content"]
    print(f"Model {config['model']} answered {answer!r} in {time.monotonic() - started:.2f}s via {config['openai_base_url']}")
    return 0


def _rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    path = db_path(config)
    if not path.exists():
        return []
    con = db.connect(path)
    try:
        found: list[dict[str, Any]] = []
        for harness in config["harnesses"]:
            found.extend(dict(row) for row in db.rows(con, harness["label"], config["split"]))
        return found
    finally:
        con.close()


def _duration(row: dict[str, Any]) -> float | None:
    start, end = _parse_time(row["started_at"]), _parse_time(row["finished_at"])
    return (end - start).total_seconds() if start and end else None


def status(config: dict[str, Any]) -> int:
    rows = _rows(config)
    if not rows:
        print(f"No executions registered for these harnesses on split {config['split']!r}.")
        return 0
    for row in rows:
        print(f"{row['execution_id']}  {row['label']:<18}  {row['status']:<16}  {row['difficulty']:<7}  {row['task']}  {row['error'] or ''}")
    print()
    for harness in config["harnesses"]:
        counts = Counter(row["status"] for row in rows if row["label"] == harness["label"])
        print(f"{harness['label']}: " + (", ".join(f"{name} {count}" for name, count in sorted(counts.items())) or "nothing yet"))
    return 0


def summarize_harness(harness: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    levels: dict[str, dict[str, Any]] = {}
    for row in rows:
        level = levels.setdefault(row["difficulty"], {"tasks": 0, "passed": 0})
        level["tasks"] += 1
        level["passed"] += row["status"] == "passed"
    for level in levels.values():
        level["accuracy"] = round(level["passed"] / level["tasks"], 4)
    durations = [d for d in (_duration(row) for row in rows if row["status"] in COMPLETED_STATUSES) if d is not None]
    costs, tokens = [], []
    outcomes: Counter = Counter()
    upstream_native = upstream_derived = discarded = unmeasured = estimated_input = 0
    usage_unavailable = usage_incomplete = 0
    for row in rows:
        result = json.loads(row["result_json"]) if row["result_json"] else {}
        usage = result.get("usage") or {}
        if isinstance(usage.get("cost_usd"), (int, float)):
            costs.append(usage["cost_usd"])
        if usage:
            tokens.append(int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0))
        timed_out = bool((result.get("agent") or {}).get("timed_out"))
        agent_exit = (result.get("agent") or {}).get("exit_code")
        if (result.get("agent_failure") or {}).get("kind") == "upstream_api_error":
            upstream_native += 1
        elif (
            row["status"] == "benchmark_failed" and agent_exit not in (0, None) and not timed_out
            and row.get("artifact_dir") and upstream_failure(Path(row["artifact_dir"]))
        ):
            upstream_derived += 1
            unmeasured += 1
        # a run stopped mid-way (deadline or operator) before its usage was recorded spent something unknown
        if result.get("usage_unavailable") or (row["status"] == "interrupted" and not usage):
            usage_unavailable += 1
        elif usage.get("complete") is False:
            usage_incomplete += 1
        discarded += int(usage.get("discarded_attempts") or 0)
        unmeasured += int(usage.get("unmeasured_attempts") or 0)
        estimated_input += int(usage.get("unmeasured_input_tokens_est") or 0)
        if row["status"] == "passed":
            outcomes["passed"] += 1
        elif row["status"] == "benchmark_failed":
            outcomes["timed_out" if timed_out else "failed"] += 1
        elif row["status"] == "interrupted":
            # a run stopped by the deadline is a timeout; one stopped by the operator is neither a timeout nor a failure
            outcomes["timed_out" if str(row.get("error") or "").startswith("deadline of") else "operator_stopped"] += 1
        elif row["status"] in ("infra_error", "stale_running"):
            outcomes["infra_error"] += 1
        else:
            outcomes["unfinished"] += 1
    return {
        "label": harness["label"],
        "harness": harness["name"],
        "settings": harness["settings"],
        "tasks": len(rows),
        "passed": outcomes["passed"],
        "failed": outcomes["failed"],
        "timed_out": outcomes["timed_out"],
        "infra_error": outcomes["infra_error"],
        "operator_stopped": outcomes["operator_stopped"],
        "unfinished": outcomes["unfinished"],
        "accuracy": round(outcomes["passed"] / len(rows), 4) if rows else None,
        "by_level": levels,
        "mean_task_seconds": round(mean(durations), 1) if durations else None,
        "tokens": sum(tokens) if tokens else None,
        "cost_usd": round(sum(costs), 4) if costs else None,
        "upstream_error_runs": upstream_native,
        "upstream_error_runs_derived": upstream_derived,
        "discarded_attempts": discarded,
        "unmeasured_attempts": unmeasured,
        "unmeasured_input_tokens_est": estimated_input,
        "usage_unavailable_runs": usage_unavailable,
        "usage_incomplete_runs": usage_incomplete,
        "cost_is_lower_bound": unmeasured > 0 or usage_unavailable > 0 or usage_incomplete > 0,
    }


def summarize(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    starts = [t for t in (_parse_time(row["started_at"]) for row in rows) if t]
    ends = [t for t in (_parse_time(row["finished_at"]) for row in rows) if t]
    finished = sum(1 for row in rows if row["finished_at"])
    completed = sum(1 for row in rows if row["status"] in COMPLETED_STATUSES)
    wall_seconds = (max(ends) - min(starts)).total_seconds() if starts and ends else None
    hours = wall_seconds / 3600 if wall_seconds else None
    return {
        "taskset": config["taskset"],
        "split": config["split"],
        "model": config["model"],
        "benchmark_slots": config["benchmark_slots"],
        "harnesses": [
            summarize_harness(harness, [row for row in rows if row["label"] == harness["label"]])
            for harness in config["harnesses"]
        ],
        "executions": len(rows),
        "completed": completed,
        "wall_seconds": round(wall_seconds, 1) if wall_seconds else None,
        "tasks_per_hour": round(completed / hours, 2) if hours else None,
        "finished_per_hour": round(finished / hours, 2) if hours else None,
    }


def _record(row: dict[str, Any]) -> dict[str, Any]:
    if row["result_json"]:
        return json.loads(row["result_json"])
    return {key: row[key] for key in ("execution_id", "label", "task", "difficulty", "status", "error")}


def report(config: dict[str, Any], output: str | None) -> int:
    rows = _rows(config)
    summary = summarize(config, rows)
    run_stats = {key: value for key, value in summary.items() if key != "harnesses"}
    for harness, harness_summary in zip(config["harnesses"], summary["harnesses"]):
        target = harness_dir(config, harness) / f"report.{config['split']}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        records = [_record(row) for row in rows if row["label"] == harness["label"]]
        target.write_text(json.dumps({"summary": harness_summary, "run": run_stats, "executions": records}, indent=2) + "\n", encoding="utf-8")
    if output:
        Path(output).write_text(json.dumps({"summary": summary, "executions": [_record(row) for row in rows]}, indent=2) + "\n", encoding="utf-8")
    print(f"limits: agent {config['agent_timeout_seconds']} s, execution deadline {config['task_deadline_seconds']} s, "
          f"{config['benchmark_slots']} agents at once")
    for harness in config["harnesses"]:
        print(f"  {harness['label']}: harness {harness['name']}, settings {json.dumps(harness['settings'])}")
    if config.get("experiment_note"):
        print(f"note: {config['experiment_note']}")
    print()
    print(f"{'harness':<20}{'tasks':>6}{'passed':>8}{'failed':>8}{'timeout':>9}{'infra':>7}{'stopped':>9}{'accuracy':>10}{'mean s':>9}{'tokens':>10}{'cost $':>10}")
    for item in summary["harnesses"]:
        accuracy = f"{item['accuracy']:.0%}" if item["accuracy"] is not None else "-"
        mean_s = f"{item['mean_task_seconds']:.0f}" if item["mean_task_seconds"] is not None else "-"
        tokens = f"{item['tokens']}" if item["tokens"] is not None else "-"
        cost = (f"{item['cost_usd']:.2f}" + ("+" if item["cost_is_lower_bound"] else "")) if item["cost_usd"] is not None else "-"
        print(f"{item['label']:<20}{item['tasks']:>6}{item['passed']:>8}{item['failed']:>8}{item['timed_out']:>9}"
              f"{item['infra_error']:>7}{item['operator_stopped']:>9}{accuracy:>10}{mean_s:>9}{tokens:>10}{cost:>10}")
    for item in summary["harnesses"]:
        if item["upstream_error_runs"] or item["upstream_error_runs_derived"]:
            parts = []
            if item["upstream_error_runs"]:
                parts.append(f"{item['upstream_error_runs']} recorded as infra errors")
            if item["upstream_error_runs_derived"]:
                parts.append(f"{item['upstream_error_runs_derived']} still counted under 'failed' but ended on an upstream API error "
                             "(derived from stored logs; the recorded status is unchanged; the failed request's usage was not recorded)")
            print(f"{item['label']}: upstream API errors: " + "; ".join(parts) + f". The denominator stays {item['tasks']}.")
        if item["discarded_attempts"]:
            print(f"{item['label']}: {item['discarded_attempts']} discarded upstream attempt(s) in total.")
        if item["operator_stopped"]:
            print(f"{item['label']}: {item['operator_stopped']} run(s) were stopped by the operator before finishing (see their stop reason "
                  f"in 'status'). They are not timeouts or failures of the harness; they stay in the denominator of {item['tasks']}.")
        if item["usage_unavailable_runs"] or item["usage_incomplete_runs"]:
            print(f"{item['label']}: usage was not recorded for {item['usage_unavailable_runs']} run(s) (unreadable, or the run was stopped "
                  f"before it was saved) and was incomplete (requests still in flight) for {item['usage_incomplete_runs']}; "
                  "their spend is not in the cost total.")
        if item["cost_is_lower_bound"]:
            reasons = []
            if item["unmeasured_attempts"]:
                reasons.append(f"{item['unmeasured_attempts']} failed upstream attempt(s) returned no usage data (input tokens estimated at "
                               f"{item['unmeasured_input_tokens_est']} where the shim could estimate, and not included in the cost)")
            if item["usage_unavailable_runs"] or item["usage_incomplete_runs"]:
                reasons.append("some runs have no complete usage record (see the note above)")
            print(f"{item['label']}: cost is a lower bound (+): " + "; ".join(reasons) + ".")
    print(f"\nexecutions {summary['executions']}, completed {summary['completed']}, wall {summary['wall_seconds']} s, "
          f"tasks/hour {summary['tasks_per_hour']}, slots {summary['benchmark_slots']}")
    return 0


def _is_stale(row: dict[str, Any], stale_minutes: float) -> bool:
    started = _parse_time(row["started_at"])
    return started is None or (datetime.now(timezone.utc) - started).total_seconds() >= stale_minutes * 60


def _recover_from_artifacts(row: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str]:
    if not row["artifact_dir"]:
        return "stale_running", None, "running execution has no artifact_dir"
    result_path = Path(row["artifact_dir"]) / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        return str(result.get("status", "infra_error")), result, f"recovered from {result_path}"
    return "stale_running", None, f"no result.json under {row['artifact_dir']}"


def recover(config: dict[str, Any], dry_run: bool, stale_minutes: float) -> int:
    path = db_path(config)
    if not path.exists():
        print("No executions found.")
        return 0
    running = [row for row in _rows(config) if row["status"] == "running"]
    con = db.connect(path)
    changed = 0
    for row in running:
        if not _is_stale(row, stale_minutes):
            print(f"SKIP     {row['execution_id']}  running for less than {stale_minutes} minutes")
            continue
        new_status, result, detail = _recover_from_artifacts(row)
        print(f"{'DRY-RUN' if dry_run else 'UPDATE '}  {row['execution_id']}  {row['label']}  {row['task']}  running -> {new_status}  {detail}")
        if not dry_run:
            db.mark_finished(con, row["execution_id"], new_status, None, result, detail)
        changed += 1
    con.close()
    if not running:
        print("No running executions found.")
    elif dry_run:
        print(f"Dry run inspected {len(running)} running execution(s).")
    else:
        print(f"Recovered {changed} execution(s).")
    return 0


def run(config: dict[str, Any]) -> int:
    try:
        with work_lock(config):
            return _run_locked(config)
    except WorkFolderBusy as error:
        print(str(error), file=sys.stderr)
        return 1


def _run_locked(config: dict[str, Any]) -> int:
    from .worker import execute

    if preflight(config):
        print("Refusing to run until preflight passes.", file=sys.stderr)
        return 1
    clean_leftovers(config, docker.from_env())
    specs = build_specs(config)
    con = db.connect(db_path(config))
    for spec in specs:
        harness = harness_by_label(config, spec.label)
        db.register(con, spec.execution_id, config_hash(config, harness), spec.label, config["split"], spec.task, spec.difficulty)
    known = {row["execution_id"]: row["status"] for row in db.rows(con)}
    con.close()
    todo = [spec for spec in specs if known.get(spec.execution_id) not in COMPLETED_STATUSES]
    print(f"\n{len(specs)} executions: {len(specs) - len(todo)} already completed, {len(todo)} to run, "
          f"{config['benchmark_slots']} at a time")
    failures = 0
    log_path = db_path(config).parent / f"run.{config['split']}.log"
    with log_path.open("a", encoding="utf-8") as log:
        def show(spec: Any, result: dict[str, Any]) -> None:
            nonlocal failures
            failures += result["status"] in ("infra_error", "interrupted")
            stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
            detail = (result.get("error") if result["status"] == "infra_error" else None) or result.get("reason") or result.get("error") or ""
            line = f"{stamp}  {spec.label:<20}{spec.task:<22}{result['status']:<17}{result.get('wall_seconds', '-')}s  {detail}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        run_specs(config, todo, execute, on_result=show)
    return 1 if failures else 0


def _relabel(entry: Any, slots: int) -> dict[str, Any]:
    entry = {"name": entry} if isinstance(entry, str) else dict(entry)
    entry["label"] = f"{entry.get('label', entry['name'])}-k{slots}"
    return entry


def sweep(config_path: str, slots: list[int]) -> int:
    source = Path(config_path).expanduser().resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    for count in slots:
        variant = {**raw, "benchmark_slots": count, "harnesses": [_relabel(entry, count) for entry in raw["harnesses"]]}
        target = source.with_name(f"{source.stem}.k{count}.json")
        target.write_text(json.dumps(variant, indent=2) + "\n", encoding="utf-8")
        print(f"\n=== sweep: {count} at a time ({target.name})")
        prepare(load_config(target), skip_firewall=True)
        run(load_config(target))
        report(load_config(target), None)
    return 0


def viz(config: dict[str, Any]) -> int:
    from .viz import build_views

    rows = _rows(config)
    page = build_views(config, rows, summarize(config, rows))
    relative = Path(page).relative_to(config["work_root"])
    print(f"Wrote {page}")
    print(f"To view: on the VM run   python3 -m http.server 8800 --bind 127.0.0.1 -d {config['work_root']}")
    print(f"then on your laptop:     ssh -L 8800:127.0.0.1:8800 cybergym   and open http://127.0.0.1:8800/{relative}")
    return 0


def server(config: dict[str, Any], action: str) -> int:
    if action == "start":
        print(f"Submission server listening at {start_server(config)}")
    elif action == "stop":
        print("Submission server stopped." if stop_server(config) else "Submission server was not running.")
    else:
        print(json.dumps(server_status(config), indent=2))
    return 0


def shim(config: dict[str, Any], action: str) -> int:
    if action == "start":
        print(f"LLM shim listening at {start_shim(config)}")
    elif action == "stop":
        print("LLM shim stopped." if stop_shim(config) else "LLM shim was not running.")
    else:
        print(json.dumps(shim_status(config), indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="cybergym-eval")
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {}
    names = ("prepare", "preflight", "check-model", "run", "status", "report", "recover", "server", "shim", "viz", "sweep")
    for name in names:
        commands[name] = sub.add_parser(name)
        commands[name].add_argument("--config", required=True)
    commands["prepare"].add_argument("--skip-firewall", action="store_true")
    commands["prepare"].add_argument("--skip-binaries", action="store_true")
    commands["prepare"].add_argument("--keep-archive", action="store_true")
    commands["report"].add_argument("--output")
    commands["recover"].add_argument("--dry-run", action="store_true")
    commands["recover"].add_argument("--stale-minutes", type=float, default=30)
    commands["server"].add_argument("action", choices=["start", "stop", "status"])
    commands["shim"].add_argument("action", choices=["start", "stop", "status"])
    commands["sweep"].add_argument("--slots", required=True, help="comma-separated, e.g. 1,2,4,8")
    args = parser.parse_args()
    if args.command == "sweep":
        raise SystemExit(sweep(args.config, [int(value) for value in args.slots.split(",")]))
    config = load_config(args.config)
    if args.command == "prepare":
        code = prepare(config, skip_firewall=args.skip_firewall, skip_binaries=args.skip_binaries, keep_archive=args.keep_archive)
    elif args.command == "preflight":
        code = preflight(config)
    elif args.command == "check-model":
        code = check_model(config)
    elif args.command == "run":
        code = run(config)
    elif args.command == "status":
        code = status(config)
    elif args.command == "report":
        code = report(config, args.output)
    elif args.command == "recover":
        code = recover(config, args.dry_run, args.stale_minutes)
    elif args.command == "viz":
        code = viz(config)
    elif args.command == "server":
        code = server(config, args.action)
    else:
        code = shim(config, args.action)
    raise SystemExit(code)
