"""Frame profiling for the slice canvas: where a pan tick's milliseconds go.

``SliceCanvas.render`` only ever timed the numpy/PIL work it does itself, which
is a fraction of what a drag costs.  A pan tick is really four things:

1. Tk delivers a motion event; the pan handler runs ``on_view_changed``, which
   in the labeler re-projects **every** screen-space annotation item.  Motion
   events arrive faster than the 15 ms repaint debounce, so this can run six or
   eight times per painted frame -- all of it outside ``render``.
2. ``render`` composites the base + overlays into one RGB array (the only part
   the old line measured).
3. ``create_image`` merely *queues* a redraw: Tk copies the photo image to the
   window later, from the event loop's idle phase.  A 1200x900 blit plus a few
   hundred overlapping canvas items is easily tens of ms, and it is invisible
   to any timer around the composite -- which is why an "80 ms render" can
   still show as 2 fps.
4. Whatever is left of the event loop: repaints coalesced away, hover work.

A ``FrameProfiler`` times all of it in one place.  Costs paid *inside* a frame
are marked as segments (``mark``); costs paid *between* frames -- the ones that
never showed up before -- accumulate in a pending bucket (``add``/``count``)
and are reported against the frame that finally lands, with the number of
events that paid them.  Per-gesture totals are summarised on release.

Off unless ``MSSEG_CANVAS_PROFILE=1`` (or ``MSCOUPON_PROFILE=1``) is set;
``SliceCanvas`` also toggles it live on Ctrl+P.  Every entry point is a cheap
no-op while off, so the calls can sit in the hot path unguarded.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager

_ENV_VARS = ("MSSEG_CANVAS_PROFILE", "MSCOUPON_PROFILE")
_FALSE = ("", "0", "false", "no", "off")


def env_enabled():
    return any(os.environ.get(n, "").strip().lower() not in _FALSE for n in _ENV_VARS)


def _fmt_ms(v):
    return f"{v:.1f}" if v < 100 else f"{v:.0f}"


class _Frame:
    """One render pass: an ordered list of named segments plus a description of
    the frame's shape.  Repeated names (one per overlay) aggregate on report,
    so three label overlays read ``ov.gather 42.1 (3x)`` instead of three
    separate entries."""

    def __init__(self, info):
        self.info = dict(info)
        self.marks = []                       # [(name, ms)] in call order
        self.t0 = self._t = time.perf_counter()

    def mark(self, name):
        t = time.perf_counter()
        self.marks.append((name, 1e3 * (t - self._t)))
        self._t = t

    @property
    def ms(self):
        return 1e3 * (time.perf_counter() - self.t0)

    def segments(self):
        """[(name, total_ms, n)] aggregated by name, in first-seen order."""
        order, tot, cnt = [], {}, {}
        for name, ms in self.marks:
            if name not in tot:
                order.append(name)
                tot[name] = 0.0
                cnt[name] = 0
            tot[name] += ms
            cnt[name] += 1
        return [(n, tot[n], cnt[n]) for n in order]


class FrameProfiler:
    def __init__(self, label="canvas", sink=None):
        self.label = label
        self.enabled = env_enabled()
        self._sink = sink or (lambda s: print(s, flush=True))
        self._frame = None
        self._pending = {}        # name -> ms paid since the last painted frame
        self._counts = {}         # name -> times paid / running counts
        self._facts = {}          # name -> last-value fact (not summed)
        self._events = 0          # input events seen since the last frame
        self._coalesced = 0       # repaints cancelled before they ran
        self._requested = None    # when the OLDEST unserved repaint was asked for
        self._last_done = None    # end of the previous painted frame
        self._n = 0
        self._gesture = None

    # -- control -------------------------------------------------------- #
    def set_enabled(self, on):
        self.enabled = bool(on)
        self._reset()
        self._gesture = None
        self._last_done = None
        self._sink(f"[perf] {self.label} profiling {'ON' if self.enabled else 'off'}"
                   + (" -- pan/zoom to see per-frame timings" if self.enabled else ""))
        return self.enabled

    # -- between-frame bookkeeping -------------------------------------- #
    def event(self, _kind="event"):
        """An input event (a pan tick, a wheel notch, a tool move) arrived."""
        if self.enabled:
            self._events += 1

    def request(self):
        """A repaint was scheduled.  Only the OLDEST unserved request is kept:
        that timestamp is the honest start of "event -> pixels on screen"."""
        if self.enabled and self._requested is None:
            self._requested = time.perf_counter()

    def coalesced(self):
        """A scheduled repaint was cancelled by a newer one (the debounce)."""
        if self.enabled:
            self._coalesced += 1

    def add(self, name, ms):
        """Charge `ms` of work done outside a frame to the next frame's line."""
        if not self.enabled:
            return
        self._pending[name] = self._pending.get(name, 0.0) + ms
        self._counts[name] = self._counts.get(name, 0) + 1

    def count(self, name, n=1):
        """Add to a running count (times paid, pixels touched)."""
        if self.enabled:
            self._counts[name] = self._counts.get(name, 0) + int(n)

    def set(self, name, value):
        """Record a LAST-value fact (how many annotation items are on the
        canvas right now) -- summing that over the events of one frame would
        just multiply it by the number of events."""
        if self.enabled:
            self._facts[name] = value

    @contextmanager
    def span(self, name):
        """Time a block of between-frame work into the pending bucket."""
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, 1e3 * (time.perf_counter() - t0))

    # -- frames --------------------------------------------------------- #
    def mark(self, name):
        """Close the running segment of the open frame (no-op when none)."""
        if self._frame is not None:
            self._frame.mark(name)

    @contextmanager
    def frame(self, **info):
        if not self.enabled:
            yield None
            return
        fr = self._frame = _Frame(info)
        try:
            yield fr
        finally:
            self._frame = None
            self._report(fr)

    # -- gestures ------------------------------------------------------- #
    def gesture(self, name):
        """Start a drag: the per-frame lines get a summary line on release."""
        if self.enabled:
            self._gesture = {"name": name, "t0": time.perf_counter(), "frames": 0,
                             "events": 0, "coalesced": 0, "seg": {}, "worst": 0.0}

    def end_gesture(self):
        g, self._gesture = self._gesture, None
        if not self.enabled or g is None or g["frames"] == 0:
            return
        dt = time.perf_counter() - g["t0"]
        n = g["frames"]
        parts = " | ".join(f"{k} {_fmt_ms(v / n)}"
                           for k, v in sorted(g["seg"].items(), key=lambda kv: -kv[1]))
        self._sink(f"[perf] {self.label} {g['name']} gesture: {dt:.2f}s, {n} frames "
                   f"({n / dt if dt else 0:.1f} fps), {g['events']} events, "
                   f"{g['coalesced']} repaints coalesced away, worst "
                   f"{_fmt_ms(g['worst'])} ms")
        if parts:
            self._sink(f"[perf]   per-frame average: {parts}")

    # -- reporting ------------------------------------------------------ #
    def _report(self, fr):
        now = time.perf_counter()
        self._n += 1
        total = fr.ms
        segs = fr.segments()
        between = sum(self._pending.values())
        gap = None if self._last_done is None else 1e3 * (now - self._last_done)
        lat = None if self._requested is None else 1e3 * (now - self._requested)

        head = " ".join(f"{k}={v}" for k, v in fr.info.items())
        body = " | ".join(f"{n} {_fmt_ms(ms)}" + (f" ({c}x)" if c > 1 else "")
                          for n, ms, c in segs)
        self._sink(f"[perf] {self.label} #{self._n} {head}")
        self._sink(f"[perf]   frame {_fmt_ms(total)} ms: {body}")
        if self._pending:
            extra = " | ".join(
                f"{n} {_fmt_ms(ms)} ({self._counts.get(n, 1)}x)"
                for n, ms in sorted(self._pending.items(), key=lambda kv: -kv[1]))
            self._sink(f"[perf]   between frames {_fmt_ms(between)} ms: {extra}")
        facts = " ".join(f"{n}={c}" for d in (self._counts, self._facts)
                         for n, c in d.items() if n not in self._pending)
        tail = f"events={self._events} coalesced={self._coalesced}"
        if lat is not None:
            tail += f" latency(evt->pixels)={_fmt_ms(lat)} ms"
        if gap:
            tail += f" gap={_fmt_ms(gap)} ms ({1e3 / gap:.1f} fps)"
        if facts:
            tail += f" {facts}"
        self._sink(f"[perf]   {tail}")

        g = self._gesture
        if g is not None:
            g["frames"] += 1
            g["events"] += self._events
            g["coalesced"] += self._coalesced
            g["worst"] = max(g["worst"], total + between)
            for n, ms, _c in segs:
                g["seg"][n] = g["seg"].get(n, 0.0) + ms
            for n, ms in self._pending.items():
                g["seg"][n] = g["seg"].get(n, 0.0) + ms
        self._reset()
        self._last_done = now

    def _reset(self):
        self._pending.clear()
        self._counts.clear()
        self._facts.clear()
        self._events = self._coalesced = 0
        self._requested = None
