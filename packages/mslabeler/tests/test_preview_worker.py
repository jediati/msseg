"""`PreviewWorker`: supersede semantics and the pump, without Tk.

The worker only needs `after` off its root, so a fake root that records
pending callbacks lets the whole thing run in-process: the tests drive the
pump by hand and can assert on exactly what was delivered.
"""
import threading
import time

import pytest

from msseg.labeler.preview import PreviewWorker


class FakeRoot:
    """Records `after` callbacks instead of scheduling them."""

    def __init__(self):
        self.pending = []

    def after(self, _ms, fn, *args):
        self.pending.append((fn, args))
        return f"id{len(self.pending)}"

    def after_cancel(self, _id):
        pass

    def run_pending(self):
        """Run whatever is scheduled right now (a real pump re-arms itself,
        so draining until empty would spin instead of waiting)."""
        queued, self.pending = self.pending, []
        for fn, args in queued:
            fn(*args)

    def pump_until(self, done, timeout=5.0):
        """Pump on a timer until `done()`. The sleep is the point: a tight
        loop starves the worker thread of the GIL and nothing ever arrives."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            self.run_pending()
            if done():
                return True
            time.sleep(0.005)
        return False


def _worker(root, results, errors=None, log=None):
    return PreviewWorker(root, lambda tok, val: results.append((tok, val)),
                         on_error=(None if errors is None
                                   else lambda tok, msg: errors.append((tok, msg))),
                         log=log)


def test_sync_delivers_on_the_spot():
    root, out = FakeRoot(), []
    w = _worker(root, out)
    w.submit(1, lambda stop: "raster", sync=True)
    assert out == [(1, "raster")]
    assert not w.busy() and not root.pending


def test_a_result_reaches_the_callback_through_the_pump():
    root, out = FakeRoot(), []
    w = _worker(root, out)
    w.submit(7, lambda stop: "raster")
    assert root.pump_until(lambda: bool(out)), "the pump re-arms until a result lands"
    assert out == [(7, "raster")]
    assert not w.busy() and not root.pending, "and stops once it has one"


def test_a_second_submit_supersedes_the_first():
    root, out = FakeRoot(), []
    w = _worker(root, out)
    release, seen = threading.Event(), []

    def slow(stop):
        release.wait(5)
        seen.append(bool(stop()))            # the stop event reached the callable
        return "first"

    w.submit(1, slow)
    w.submit(2, lambda stop: "second", sync=True)   # supersedes 1
    assert out == [(2, "second")], "the newer result is delivered at once"
    release.set()
    assert root.pump_until(lambda: len(seen) == 1)
    root.run_pending()
    assert seen == [True], "the superseded callable is told to stop"
    assert out == [(2, "second")], "and its raster is never painted"


def test_an_exception_reaches_on_error_and_frees_the_worker():
    root, out, errs = FakeRoot(), [], []
    w = _worker(root, out, errors=errs)

    def boom(stop):
        raise ValueError("no such channel")

    w.submit(3, boom, sync=True)
    assert out == [] and errs and errs[0][0] == 3
    assert "ValueError: no such channel" in errs[0][1]
    assert not w.busy(), "a failure must not leave the spinner on"


def test_without_an_error_handler_the_failure_is_logged():
    root, out, lines = FakeRoot(), [], []
    w = _worker(root, out, log=lines.append)
    w.submit(4, lambda stop: 1 / 0, sync=True)
    assert out == [] and any("ZeroDivisionError" in m for m in lines), lines


def test_a_thread_that_dies_silently_does_not_wedge_the_worker():
    root, out, errs = FakeRoot(), [], []
    w = _worker(root, out, errors=errs)
    # Forge the state a vanished thread leaves: busy, nothing queued, and a
    # thread object that is not alive.
    w._stop, w._token = threading.Event(), 9
    w._thread = threading.Thread(target=lambda: None)
    w._thread.start()
    w._thread.join()
    w._pump()
    assert not w.busy() and errs == [(9, "the worker thread ended silently")]


def test_stop_drops_a_late_result():
    root, out = FakeRoot(), []
    w = _worker(root, out)
    release, ran = threading.Event(), threading.Event()

    def slow(stop):
        release.wait(5)
        ran.set()
        return "raster"

    w.submit(5, slow)
    w.stop()
    release.set()
    assert ran.wait(5), "the callable still finishes -- nothing is aborted"
    w._pump()
    assert out == [], "but its raster is never painted"
    assert not w.busy()
