from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from ray import cloudpickle

from .config import RunSpec
from .sandbox import SANDBOX_IMAGE, ModelAccess, Sandbox, Task
from .shim import shim_url


def main() -> None:
    job = cloudpickle.loads(sys.stdin.buffer.read())
    config, spec = job["config"], RunSpec(**job["spec"])
    output = Path(job["output"])
    agent_class = job["agent_class"]
    task = Task(id=spec.task, difficulty=spec.difficulty, time_limit_seconds=job["timeout"], deadline=job["deadline"])
    lm = ModelAccess(base_url=shim_url(config), api_key=job["key"], model=config["model"])
    box = Sandbox(
        config, spec.execution_id, Path(job["workspace"]), output,
        image=getattr(agent_class, "image", SANDBOX_IMAGE), log_path=output / "trajectory.jsonl",
    )
    error = None
    poc = None
    try:
        box.start()
        poc = agent_class(**job["settings"]).solve(task, box, lm)
    except Exception:
        error = traceback.format_exc()[-4000:]
        print(error, file=sys.stderr, flush=True)
    finally:
        box.close()
    if isinstance(poc, str):
        poc = poc.encode()
    if isinstance(poc, (bytes, bytearray)) and poc:
        (output / "poc").write_bytes(bytes(poc))
    report = {"error": error, "oom_killed": box.oom_killed, "commands": box.commands}
    (output.parent / "agent_exit.json").write_text(json.dumps(report), encoding="utf-8")


if __name__ == "__main__":
    main()
