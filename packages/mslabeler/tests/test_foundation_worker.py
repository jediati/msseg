"""GridWorker (a thread + a Tk-side pump) and the fm_deferred decorator, with a
fake root whose `after` queues callbacks for the test to run.

    pytest packages/mslabeler/tests/test_foundation_worker.py
"""
import threading
import time

import pytest

from msseg.labeler.foundation.worker import GridWorker, GridsPending, fm_deferred


class FakeRoot:
    def __init__(self):
        self.calls = []

    def after(self, _ms, fn):
        self.calls.append(fn)

    def run(self, until, limit=10.0):
        end = time.time() + limit
        while not until() and time.time() < end:
            calls, self.calls = self.calls, []
            for fn in calls:
                fn()
            time.sleep(0.005)
        return until()


def test_jobs_run_off_thread_and_deliver_on_the_pump():
    root = FakeRoot()
    done, errors, threads = {}, {}, set()
    w = GridWorker(root, lambda ck, v: done.__setitem__(ck, v),
                   lambda ck, m: errors.__setitem__(ck, m), pump_ms=1)

    def job(v):
        def f():
            threads.add(threading.current_thread().name)
            return v
        return f

    def boom():
        raise RuntimeError("no pixels")

    assert w.submit("a", job(1)) and w.submit("b", job(2)) and w.submit("c", boom)
    assert not w.submit("a", job(99)), "a queued key is not queued twice"
    assert w.pending() == 3 and w.is_pending("b")
    assert not done, "nothing is delivered before the pump runs"
    assert root.run(lambda: w.pending() == 0)
    assert done == {"a": 1, "b": 2} and "RuntimeError: no pixels" in errors["c"]
    assert threads == {"foundation"}
    assert w.done == 2 and w.failed == 1


def test_cancel_drops_queued_and_running_results():
    root = FakeRoot()
    done = []
    gate = threading.Event()
    w = GridWorker(root, lambda ck, v: done.append(ck), lambda ck, m: None, pump_ms=1)
    w.submit("slow", lambda: gate.wait(5) or "x")
    w.submit("queued", lambda: "y")
    assert w.cancel() == 2 and w.pending() == 0
    gate.set()
    time.sleep(0.1)
    root.run(lambda: False, limit=0.2)
    assert done == [], "a cancelled result was delivered"
    w.submit("after", lambda: "z")
    assert root.run(lambda: w.pending() == 0) and done == ["after"]


class Op:
    def __init__(self):
        self.calls = 0
        self.deferred = []

    def _fm_defer(self, name, retry, keys):
        self.deferred.append((name, retry, set(keys)))

    @fm_deferred
    def train(self, x, y=0):
        self.calls += 1
        if self.calls == 1:
            assert self._fm_scopes == ["all"]
            raise GridsPending({"k1", "k2"})
        return x + y

    @fm_deferred(scope="one")
    def current(self):
        return list(self._fm_scopes)


def test_deferred_swallows_and_retries_the_same_call():
    op = Op()
    assert op.train(2, y=3) is None
    (name, retry, keys), = op.deferred
    assert name == "train" and keys == {"k1", "k2"}
    assert op._fm_scopes == [], "the scope stack unwinds"
    assert retry() == 5 and op.calls == 2
    assert op.current() == ["one"]


def test_other_errors_pass_through():
    class Bad:
        _fm_defer = None

        @fm_deferred
        def go(self):
            raise ValueError("nope")
    with pytest.raises(ValueError):
        Bad().go()
    assert not issubclass(GridsPending, ValueError)
