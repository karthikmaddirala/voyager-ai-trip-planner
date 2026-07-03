"""
logutil.py
──────────
Tiny structured logging so you can follow what the backend is doing in the
terminal: which endpoint/stage ran, the AI calls it made, the external APIs it
hit, and how long each took.

Usage:
    from logutil import log, step
    log("API", "/i/route  session=ab12  stops=3")
    with step("feasibility", "judging 6 stops"):
        ...                      # prints ▶ start and ✓ end with elapsed seconds
"""

import sys
import time

# The logs use unicode glyphs (→ ★ ✓ ▶ ✗). On Windows the console defaults to
# cp1252, which can't encode them and raises UnicodeEncodeError mid-request —
# so force the stdout/stderr streams to UTF-8 (replace anything unencodable).
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def log(tag: str, msg: str = "") -> None:
    try:
        print(f"[{tag:<11}] {msg}", flush=True)
    except UnicodeEncodeError:
        # last-resort fallback if the stream still can't encode a glyph
        line = f"[{tag:<11}] {msg}"
        sys.stdout.buffer.write(line.encode("utf-8", "replace") + b"\n")
        sys.stdout.flush()


class step:
    """Context manager that logs start, end, and elapsed time for a block."""
    def __init__(self, tag: str, msg: str = ""):
        self.tag, self.msg = tag, msg

    def __enter__(self):
        self.t0 = time.time()
        log(self.tag, f"▶ {self.msg}")
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        if exc_type:
            log(self.tag, f"✗ {self.msg} FAILED after {dt:.1f}s — {exc}")
        else:
            log(self.tag, f"✓ {self.msg} ({dt:.1f}s)")
        return False
