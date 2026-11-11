from __future__ import annotations

import re
from pathlib import Path

UPSTREAM_ERROR = re.compile(
    r"The server had an error while processing your request"
    r"|internal_server_error|upstream_error|upstream unreachable|server_error"
    r"|ECONNRESET|socket hang up|ETIMEDOUT"
    r"|\b50[234]\b[^\n]{0,40}(Bad Gateway|Service Unavailable|Gateway Time-?out)"
    r"|overloaded"
    r"|Network error while contacting|upstream stream failed|upstream_stream_failed",
    re.IGNORECASE,
)
# codex.out also carries the target program's tool output, so only these system messages are trusted there
# (broad patterns such as 'server_error' or '502' would match what the target prints), and only near the end:
# a run that ends on an upstream failure has the message among its last lines.
SYSTEM_MESSAGE = re.compile(
    r"Network error while contacting|upstream stream failed|upstream_stream_failed"
    r"|The server had an error while processing your request",
    re.IGNORECASE,
)
CODEX_OUT_LAST_LINES = 50
TAIL_BYTES = 40000


def _tail(path: Path) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - TAIL_BYTES))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def upstream_failure(folder: Path) -> str | None:
    candidates = sorted((folder / "output").glob("*.err")) + [folder / "agent.log"]
    for path in candidates:
        for line in _tail(path).splitlines():
            if UPSTREAM_ERROR.search(line):
                return line.strip()[:200]
    last = [line for line in _tail(folder / "output" / "codex.out").splitlines() if line.strip()][-CODEX_OUT_LAST_LINES:]
    for line in last:
        if SYSTEM_MESSAGE.search(line):
            return line.strip()[:200]
    return None
