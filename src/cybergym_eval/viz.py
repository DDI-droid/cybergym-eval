from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import execution_dir, harness_by_label, taskset_dir

COLORS = {
    "passed": "#2e7d32",
    "benchmark_failed": "#8d8d8d",
    "interrupted": "#ef6c00",
    "infra_error": "#c62828",
    "running": "#1565c0",
    "pending": "#b0bec5",
    "stale_running": "#6d4c41",
}
LABELS = {"benchmark_failed": "failed", "interrupted": "timed out", "infra_error": "infra error"}

STYLE = """
:root { --bg: #ffffff; --fg: #1f2328; --muted: #656d76; --line: #d0d7de; --card: #f6f8fa; }
@media (prefers-color-scheme: dark) { :root { --bg: #0d1117; --fg: #e6edf3; --muted: #8b949e; --line: #30363d; --card: #161b22; } }
body { margin: 0; background: var(--bg); color: var(--fg); font: 14px/1.5 -apple-system, "Segoe UI", system-ui, sans-serif; }
main { max-width: 1200px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 16px; margin: 32px 0 8px; }
.muted { color: var(--muted); }
table { border-collapse: collapse; width: 100%; }
th, td { border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: left; white-space: nowrap; }
th { font-weight: 600; color: var(--muted); }
td.num, th.num { text-align: right; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 10px; color: #fff; font-size: 12px; text-decoration: none; }
.grid td { text-align: center; }
.scroll { overflow-x: auto; }
.step { border: 1px solid var(--line); border-radius: 6px; margin: 8px 0; background: var(--card); }
.step header { padding: 6px 10px; display: flex; gap: 12px; border-bottom: 1px solid var(--line); }
.step pre, details pre { margin: 0; padding: 8px 10px; white-space: pre-wrap; word-break: break-word; font: 12px/1.45 ui-monospace, Menlo, Consolas, monospace; }
details { margin: 8px 0; } summary { cursor: pointer; color: var(--muted); }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 2px 16px; } dt { color: var(--muted); } dd { margin: 0; }
"""


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _badge(status: str, href: str | None = None) -> str:
    text = _esc(LABELS.get(status, status))
    color = COLORS.get(status, "#555")
    if href:
        return f'<a class="badge" style="background:{color}" href="{_esc(href)}">{text}</a>'
    return f'<span class="badge" style="background:{color}">{text}</span>'


def _page(title: str, body: str) -> str:
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_esc(title)}</title><style>{STYLE}</style></head><body><main>{body}</main></body></html>"
    )


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def trajectory_page(folder: Path, row: dict[str, Any]) -> Path:
    result_path = folder / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    steps_path = folder / "output" / "trajectory.jsonl"
    steps = []
    if steps_path.exists():
        for line in steps_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                try:
                    steps.append(json.loads(line))
                except ValueError:
                    steps.append({"type": "unparsed", "text": line})
    log_path = folder / "agent.log"
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    final = result.get("final") or {}
    agent = result.get("agent") or {}
    usage = result.get("usage") or {}
    facts = [
        ("status", _badge(row["status"])),
        ("task", _esc(row["task"]) + f' <span class="muted">({_esc(row["difficulty"])})</span>'),
        ("harness", _esc(result.get("harness")) + f' <span class="muted">as {_esc(row["label"])}</span>'),
        ("why", _esc(result.get("reason") or result.get("error") or "")),
        ("final PoC", _esc(f"vulnerable exit {final.get('vul_exit_code')}, fixed exit {final.get('fix_exit_code')}, "
                            f"{final.get('poc_length')} bytes") if final else "none"),
        ("submissions", _esc(result.get("submissions"))),
        ("any submission would pass", _esc(result.get("any_submission_passed"))),
        ("agent", _esc(f"{agent.get('seconds')} s, exit {agent.get('exit_code')}, timed out {agent.get('timed_out')}, "
                        f"out of memory {agent.get('oom_killed')}") if agent else ""),
        ("model use", _esc(f"{usage.get('requests')} requests, {usage.get('input_tokens')} in / {usage.get('output_tokens')} out tokens, "
                            f"{usage.get('throttled')} throttled, {usage.get('model_seconds')} s") if usage else ""),
        ("times", _esc(f"grading {result.get('grading_seconds')} s, "
                        f"total {result.get('wall_seconds')} s")),
        ("code", _esc((result.get("code_sha256") or "")[:16])),
    ]
    rendered = []
    for step in steps:
        rest = {k: v for k, v in step.items() if k not in ("t", "step", "type")}
        text = json.dumps(rest, indent=2, ensure_ascii=False)
        body = f"<pre>{_esc(text)}</pre>" if len(text) < 4000 else f"<details><summary>{len(text)} characters</summary><pre>{_esc(text)}</pre></details>"
        when = f"+{step['t']} s" if "t" in step else f"step {step.get('step', '')}"
        rendered.append(f'<div class="step"><header><b>{_esc(step.get("type"))}</b>'
                        f'<span class="muted">{_esc(when)}</span></header>{body}</div>')
    body = (
        f"<p><a href=\"../../view.{_esc(row['split'])}.html\">&larr; run</a></p>"
        f"<h1>{_esc(row['task'])} &middot; {_esc(row['label'])}</h1>"
        "<dl>" + "".join(f"<dt>{name}</dt><dd>{value}</dd>" for name, value in facts) + "</dl>"
        f"<h2>Trajectory ({len(steps)} steps)</h2>" + ("".join(rendered) or '<p class="muted">No trajectory.jsonl was written.</p>')
        + f"<h2>Container log</h2><details><summary>{len(log)} characters</summary><pre>{_esc(log[-200000:])}</pre></details>"
    )
    target = folder / "trajectory.html"
    target.write_text(_page(f"{row['task']} {row['label']}", body), encoding="utf-8")
    return target


def _timeline(rows: list[dict[str, Any]]) -> str:
    spans = []
    for row in rows:
        start, end = _parse(row["started_at"]), _parse(row["finished_at"])
        if start:
            spans.append((start, end or datetime.now(timezone.utc), row))
    if not spans:
        return '<p class="muted">Nothing has run yet.</p>'
    spans.sort(key=lambda item: item[0])
    origin = spans[0][0]
    total = max((end - origin).total_seconds() for _, end, _ in spans) or 1
    lanes: list[datetime] = []
    placed = []
    for start, end, row in spans:
        lane = next((index for index, free_at in enumerate(lanes) if free_at <= start), None)
        if lane is None:
            lanes.append(end)
            lane = len(lanes) - 1
        else:
            lanes[lane] = end
        placed.append((lane, start, end, row))
    width, lane_height, left = 1000, 18, 40
    height = len(lanes) * lane_height + 30
    bars = []
    for lane, start, end, row in placed:
        x = left + (start - origin).total_seconds() / total * (width - left - 10)
        w = max(2, (end - start).total_seconds() / total * (width - left - 10))
        y = 4 + lane * lane_height
        tip = f"{row['label']} · {row['task']} · {LABELS.get(row['status'], row['status'])} · {(end - start).total_seconds():.0f} s"
        bars.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{lane_height - 4}" rx="2" '
                    f'fill="{COLORS.get(row["status"], "#555")}"><title>{_esc(tip)}</title></rect>')
    ticks = []
    for index in range(6):
        seconds = total * index / 5
        x = left + seconds / total * (width - left - 10)
        ticks.append(f'<text x="{x:.1f}" y="{height - 6}" font-size="11" fill="currentColor" text-anchor="middle">{seconds / 60:.0f}m</text>')
    labels = "".join(
        f'<text x="4" y="{4 + lane * lane_height + 12}" font-size="11" fill="currentColor">{lane + 1}</text>' for lane in range(len(lanes))
    )
    return (f'<div class="scroll"><svg viewBox="0 0 {width} {height}" width="100%" role="img" '
            f'aria-label="Timeline of executions">{labels}{"".join(bars)}{"".join(ticks)}</svg></div>'
            f'<p class="muted">Each row is a slot; each bar is one execution (hover for details). '
            f'{len(lanes)} ran at the same time at most.</p>')


def build_views(config: dict[str, Any], rows: list[dict[str, Any]], summary: dict[str, Any]) -> Path:
    for row in rows:
        row["split"] = config["split"]
        folder = execution_dir(config, harness_by_label(config, row["label"]), row["task"])
        if folder.exists():
            trajectory_page(folder, row)
    header = (
        f"<h1>{_esc(config['taskset'])} &middot; {_esc(config['split'])}</h1>"
        f"<p class=\"muted\">model {_esc(config['model'])} &middot; {_esc(summary['executions'])} executions &middot; "
        f"{_esc(config['benchmark_slots'])} at a time &middot; generated "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>"
    )
    result_rows = []
    for item in summary["harnesses"]:
        accuracy = f"{item['accuracy']:.0%}" if item["accuracy"] is not None else "-"
        cost = f"{item['cost_usd']:.2f}" if item["cost_usd"] is not None else "-"
        result_rows.append(
            f"<tr><td>{_esc(item['label'])} <span class=\"muted\">{_esc(item['harness'])}</span></td>"
            f"<td class=\"num\">{item['tasks']}</td><td class=\"num\">{item['passed']}</td><td class=\"num\">{item['failed']}</td>"
            f"<td class=\"num\">{item['timed_out']}</td><td class=\"num\">{item['infra_error']}</td><td class=\"num\">{accuracy}</td>"
            f"<td class=\"num\">{_esc(item['mean_task_seconds'] or '-')}</td><td class=\"num\">{_esc(item['tokens'] or '-')}</td>"
            f"<td class=\"num\">{cost}</td></tr>"
        )
    results = (
        "<h2>Results</h2><div class=\"scroll\"><table><tr><th>harness</th><th class=\"num\">tasks</th><th class=\"num\">passed</th>"
        "<th class=\"num\">failed</th><th class=\"num\">timed out</th><th class=\"num\">infra errors</th><th class=\"num\">accuracy</th>"
        "<th class=\"num\">mean s / task</th><th class=\"num\">tokens</th><th class=\"num\">cost $</th></tr>"
        + "".join(result_rows) + "</table></div>"
    )
    labels = [harness["label"] for harness in config["harnesses"]]
    by_key = {(row["task"], row["label"]): row for row in rows}
    tasks = list(dict.fromkeys(row["task"] for row in rows))
    grid_rows = []
    for task in tasks:
        cells = []
        for label in labels:
            row = by_key.get((task, label))
            if row is None:
                cells.append("<td class=\"muted\">-</td>")
                continue
            href = f"{label}/{task.replace(':', '_')}/trajectory.html"
            cells.append(f"<td>{_badge(row['status'], href)}</td>")
        grid_rows.append(f"<tr><td>{_esc(task)}</td>{''.join(cells)}</tr>")
    grid = (
        "<h2>Tasks &times; harnesses</h2><p class=\"muted\">Click a cell for that execution's trajectory.</p>"
        "<div class=\"scroll\"><table class=\"grid\"><tr><th>task</th>"
        + "".join(f"<th>{_esc(label)}</th>" for label in labels) + "</tr>" + "".join(grid_rows) + "</table></div>"
    )
    timeline = "<h2>Timeline</h2>" + _timeline(rows)
    target = taskset_dir(config) / f"view.{config['split']}.html"
    target.write_text(_page(f"{config['taskset']} {config['split']}", header + results + grid + timeline), encoding="utf-8")
    return target
