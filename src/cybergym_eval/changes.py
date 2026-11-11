from __future__ import annotations

from pathlib import Path

CHANGE_RECORD = Path.home() / ".local" / "state" / "cybergym-eval"


def record_change(name: str, value: str | None = None) -> None:
    CHANGE_RECORD.mkdir(parents=True, exist_ok=True)
    path = CHANGE_RECORD / name
    if value is None:
        path.touch()
        return
    known = path.read_text(encoding="utf-8").split() if path.exists() else []
    if value not in known:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(value + "\n")
