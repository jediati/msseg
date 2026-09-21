"""One background compute at a time, newest wins, drained by a Tk pump.

The filter chain that feeds the topology computation is judged by LOOKING at
its output, so a parameter edit has to repaint the image. That recompute is
not cheap -- a GMM normalize is seconds on a 3232^2 slice, a blur at a large
sigma not much less -- and doing it on the Tk thread freezes the window on
every keystroke, which is exactly the loop the edit is in.

So it runs here instead. The shape is the Optimize search's worker and pump
(`classifier.py`), minus everything specific to a search: a daemon thread, a
`queue.Queue` the Tk thread drains on a `root.after` timer, and a
`threading.Event` the callable can consult between stages.

Two things this deliberately does NOT do:

* It does not abort. A `filter_slice` already inside the extension cannot be
  interrupted, so a superseded 5 s stage finishes. What the stop event buys is
  bailing out BETWEEN stages of a chain; a late result is then dropped rather
  than painted. Call it supersede, not cancel.
* It does not touch the caller's caches. The callable returns arrays and the
  pump stores them, so every cache stays single-threaded (the raster LRUs are
  bare OrderedDicts, and even a read mutates them).
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk


class PreviewWorker:
    """`submit(token, fn)` runs `fn(should_stop)` off the Tk thread and
    delivers its return value to `on_result(token, value)` ON the Tk thread.

    A second submit supersedes the first: the older stop event is set and its
    result, whenever it turns up, is dropped here rather than handed on -- so
    `on_result` always means "this is the current preview". An exception
    reaches `on_error(token, message)` (or `log`, when no handler was given)
    and never escapes the thread. `submit(..., sync=True)` runs inline and
    pumps once, which is how the selftests drive it without threads or a live
    event loop.
    """

    def __init__(self, root, on_result, on_error=None, log=None,
                 pump_ms=60, name="preview"):
        self.root = root
        self.on_result = on_result
        self.on_error = on_error
        self.log = log or (lambda _m: None)
        self.pump_ms = int(pump_ms)
        self.name = str(name)
        self._queue = queue.Queue()
        self._stop = None          # the live submission's stop event
        self._thread = None
        self._token = None         # the live submission's token
        self._armed = False        # a pump callback is pending

    # -- submitting ------------------------------------------------------- #
    def submit(self, token, fn, sync=False):
        """Run `fn(should_stop)` for `token`, superseding anything in flight."""
        self.stop()
        stop = threading.Event()
        self._stop, self._token = stop, token

        def work():
            try:
                out = fn(stop.is_set)
            except Exception as exc:                # reported, never raised off-thread
                self._queue.put((token, "error", f"{type(exc).__name__}: {exc}"))
                return
            self._queue.put((token, "done", out))

        if sync:
            work()
            self._pump()
            return
        t = threading.Thread(target=work, name=self.name, daemon=True)
        self._thread = t
        t.start()
        self._arm()

    def stop(self):
        """Supersede whatever is running: set its event and forget the token,
        which is what makes the pump drop the result if it still turns up."""
        if self._stop is not None:
            self._stop.set()
        self._stop, self._thread, self._token = None, None, None

    def busy(self):
        """True while a submission is outstanding (its result not yet drained)."""
        return self._stop is not None

    def token(self):
        return self._token

    # -- draining --------------------------------------------------------- #
    def _arm(self):
        if self._armed:
            return
        try:
            self.root.after(self.pump_ms, self._pump)
            self._armed = True
        except tk.TclError:
            pass

    def _pump(self):
        self._armed = False
        done = False
        try:
            while True:
                token, kind, payload = self._queue.get_nowait()
                if token != self._token:
                    # Superseded: a newer submit is what the caller is waiting
                    # for, and this raster was computed from parameters that
                    # have already moved.
                    if kind == "error":
                        self.log(f"preview superseded, and failed: {payload}")
                    continue
                self._stop, self._thread, self._token = None, None, None
                done = True
                if kind == "error":
                    if self.on_error is not None:
                        self.on_error(token, payload)
                    else:
                        self.log(f"preview failed: {payload}")
                else:
                    self.on_result(token, payload)
        except queue.Empty:
            pass
        # A thread that died without queueing anything would otherwise leave
        # the worker "busy" for ever, and with it the caller's spinner.
        t = self._thread
        if not done and t is not None and not t.is_alive() and self._queue.empty():
            tok = self._token
            self._stop, self._thread, self._token = None, None, None
            self.log("preview: the worker thread ended silently")
            if self.on_error is not None:
                self.on_error(tok, "the worker thread ended silently")
            return
        if self.busy():
            self._arm()
