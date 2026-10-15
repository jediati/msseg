"""Token grids off the Tk thread.

``GridWorker`` is one daemon thread over a FIFO of grid jobs (read the item's
pixels, run the backbone, stitch the grid); results are drained on the Tk
thread by a ``root.after`` pump and handed to ``on_done(ck, (grid, stride))``
or ``on_error(ck, message)`` -- the shape of ``preview.PreviewWorker``, but a
queue rather than a supersede-the-last slot, because a Train needs every
item's grid and none of them replaces another. ``cancel()`` drops what is
queued; the job running finishes and its result is discarded.

``GridsPending`` is how a model operation learns that a grid is not cached
yet: the labeler enqueues what it needs, raises, and ``fm_deferred`` -- the
decorator on the operation's entry point -- swallows it and re-runs the
operation once those grids have landed (``FoundationMixin._fm_defer``). It is
deliberately NOT a ValueError, so the handlers that turn a ValueError into
"this item cannot be computed" let it through.
"""
from __future__ import annotations

import functools
import queue
import threading
from typing import Any, Callable, Dict, Optional


class GridsPending(Exception):
    """Raised inside a deferred operation when a token grid is being extracted."""

    def __init__(self, keys=()):
        super().__init__(f"{len(keys)} foundation grid(s) are being extracted")
        self.keys = set(keys)


def fm_deferred(fn=None, *, scope="all"):
    """Decorate a model operation's entry point: a ``GridsPending`` raised
    inside it is swallowed (the call returns None) and the same call is
    re-run once the grids it waited for are cached. `scope` says what a miss
    enqueues: ``"all"`` every ready item (Train, Classify all, Evaluate) or
    ``"one"`` only the missing item (classify what is on screen)."""
    def wrap(f):
        name = f.__name__

        @functools.wraps(f)
        def wrapper(self, *a, **k):
            stack = self.__dict__.setdefault("_fm_scopes", [])
            stack.append(scope)
            try:
                return f(self, *a, **k)
            except GridsPending as pending:
                self._fm_defer(name, lambda: getattr(self, name)(*a, **k), pending.keys)
                return None
            finally:
                stack.pop()
        return wrapper
    return wrap(fn) if fn is not None else wrap


class GridWorker:
    def __init__(self, root, on_done: Callable[[str, Any], None],
                 on_error: Callable[[str, str], None], log=None, pump_ms: int = 80,
                 name: str = "foundation"):
        self.root = root
        self.on_done = on_done
        self.on_error = on_error
        self.log = log or (lambda _m: None)
        self.pump_ms = int(pump_ms)
        self.name = name
        self._jobs: "queue.Queue" = queue.Queue()
        self._results: "queue.Queue" = queue.Queue()
        self._inflight: Dict[str, str] = {}     # ck -> label, Tk-side bookkeeping
        self._gen = 0                           # bumped by cancel(); stale results drop
        self._thread: Optional[threading.Thread] = None
        self._armed = False
        self.done = self.failed = 0

    # -- submitting (Tk thread) ------------------------------------------- #
    def submit(self, ck: str, fn: Callable[[], Any], label: str = "") -> bool:
        """Queue `fn()` for cache key `ck`; False when it is already queued."""
        if ck in self._inflight:
            return False
        self._inflight[ck] = label
        self._jobs.put((self._gen, ck, fn))
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
            self._thread.start()
        self._arm()
        return True

    def cancel(self) -> int:
        """Drop every queued job (the running one finishes, unreported)."""
        self._gen += 1
        n = len(self._inflight)
        try:
            while True:
                self._jobs.get_nowait()
        except queue.Empty:
            pass
        self._inflight.clear()
        return n

    def pending(self) -> int:
        return len(self._inflight)

    def is_pending(self, ck: str) -> bool:
        return ck in self._inflight

    # -- the thread ------------------------------------------------------- #
    def _run(self):
        while True:
            gen, ck, fn = self._jobs.get()
            try:
                out = fn()
            except Exception as exc:               # reported, never raised off-thread
                self._results.put((gen, ck, "error", f"{type(exc).__name__}: {exc}"))
            else:
                self._results.put((gen, ck, "done", out))

    # -- the pump (Tk thread) --------------------------------------------- #
    def _arm(self):
        if not self._armed:
            self._armed = True
            try:
                self.root.after(self.pump_ms, self._pump)
            except Exception:                        # a destroyed root
                self._armed = False

    def _pump(self):
        self._armed = False
        while True:
            try:
                gen, ck, kind, val = self._results.get_nowait()
            except queue.Empty:
                break
            if gen != self._gen:
                continue                             # cancelled after it started
            self._inflight.pop(ck, None)
            if kind == "done":
                self.done += 1
                self.on_done(ck, val)
            else:
                self.failed += 1
                self.on_error(ck, val)
        if self._inflight:
            self._arm()
