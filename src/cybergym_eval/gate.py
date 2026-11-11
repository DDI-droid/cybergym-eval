"""A file in a run's output folder that the agent cannot write until a crash is confirmed.

The agent runs as the same user as the worker, so nothing inside the container can be trusted to
hold a lock. The lock is therefore made of things the agent has no authority over:

* The file is created on the host and made root-owned (mode 0644) by a short root container.
  The agent is not the owner, so opening it for writing, chmod and chown all fail.
* The file is bind-mounted over its path in the container. A mount point cannot be unlinked or
  renamed over, so the file cannot be replaced, and the directory around it stays writable for logs.
* The worker, which is outside the container, asks the submission server (with the host API key,
  which the container never sees) whether any submission of this run crashed the vulnerable build.
  Only then does it hand the file to the agent's user, with `docker exec --user root chown`.

Nothing is polled to delete anything. A write before the crash is refused by the kernel at open().
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable


class OutputGate:
    def __init__(
        self, client: Any, image: str, output: Path, name: str, confirmed: Callable[[], dict[str, Any] | None],
        uid: int, gid: int, interval: float = 1.0,
    ):
        self.client, self.image = client, image
        self.path = Path(output) / name
        self.name = name
        self.confirmed = confirmed
        self.uid, self.gid = uid, gid
        self.interval = interval
        self.events: dict[str, Any] = {}
        self._stop = threading.Event()
        self._closed = False
        self._thread: threading.Thread | None = None
        self._container: Any = None

    def lock(self) -> None:
        self.path.unlink(missing_ok=True)
        self.path.write_bytes(b"")
        self.path.chmod(0o644)
        self.client.containers.run(
            self.image, ["chown", "0:0", f"/gate/{self.name}"], user="0", network_mode="none", remove=True,
            volumes={str(self.path.parent): {"bind": "/gate", "mode": "rw"}},
        )
        self.events["locked_at"] = time.time()

    def watch(self, container: Any) -> None:
        self._container = container
        self._thread = threading.Thread(target=self._loop, name=f"gate-{self.name}", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                verdict = self.confirmed()
            except Exception as error:
                self.events["last_probe_error"] = f"{type(error).__name__}: {error}"[:300]
                continue
            if self._stop.is_set():
                return  # closed while the query was running: never unlock after shutdown began
            if verdict and self._unlock(verdict):
                return

    def _unlock(self, verdict: dict[str, Any]) -> bool:
        target = f"/output/{self.name}"
        opening = time.time()
        try:
            code, output = self._container.exec_run(["chown", f"{self.uid}:{self.gid}", target], user="0")
            if code != 0:
                raise RuntimeError(f"chown exited {code}: {(output or b'').decode(errors='replace')[:200]}")
            code, _ = self._container.exec_run(["test", "-w", target], user=f"{self.uid}:{self.gid}")
            if code != 0:
                raise RuntimeError("the file is still not writable by the agent after chown")
        except Exception as error:
            self.events["last_unlock_error"] = f"{type(error).__name__}: {error}"[:300]
            return False
        # opening_at is taken before the chown: the agent may write the moment the chown returns.
        self.events["opening_at"] = opening
        self.events["opened_at"] = time.time()
        self.events["confirmed_by"] = verdict
        return True

    def close(self) -> dict[str, Any]:
        if self._closed:
            return self.events
        self._closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
        try:
            info = self.path.stat()
        except FileNotFoundError:
            info = None
        opening = self.events.get("opening_at")
        self.events["opened"] = opening is not None
        self.events["final_bytes"] = info.st_size if info else 0
        if info and info.st_size > 0:
            # A write before the unlock is refused by the kernel, so this is a consistency check on the
            # record, not the mechanism. The tolerance covers the kernel's coarse file-timestamp clock.
            self.events["final_written_after_open"] = opening is not None and info.st_mtime >= opening - 0.05
        if info and info.st_size == 0:
            self.path.unlink(missing_ok=True)
        try:
            (self.path.parent.parent / "gate.json").write_text(json.dumps(self.events, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass
        return self.events
