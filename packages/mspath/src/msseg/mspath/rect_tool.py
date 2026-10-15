"""A one-shot rectangle drag on the slide canvas: the "Draw rect" place tool.

Armed by the ROI section's button, it stands in for the canvas's drawing tool
until one rectangle is dragged (or Escape / a second click on the button
cancels), then puts the previous tool back and hands the rect -- in SLIDE
pixels, the canvas's own coordinates -- to a callback. Attribute reads it does
not know fall through to the tool it replaced, so code that peeks at
``viewer.tool`` (an in-flight trace, ``cancel``) keeps working while it is
armed. Middle- and right-drag still pan, as with every tool.
"""
from __future__ import annotations

TAG = "placerect"
COLOR = "#ffd84a"


class RectPlaceTool:
    def __init__(self, viewer, on_rect, on_cancel=None):
        self.viewer = viewer
        self.on_rect = on_rect           # callback(x0, y0, x1, y1) in slide px
        self.on_cancel = on_cancel       # callback() when disarmed without a rect
        self.prev = viewer.tool
        self._start = None
        viewer.tool = self

    def __getattr__(self, name):         # only called for names not found here
        prev = self.__dict__.get("prev")
        if prev is None:
            raise AttributeError(name)
        return getattr(prev, name)

    @property
    def armed(self):
        return self.viewer.tool is self

    def _disarm(self):
        self.viewer.canvas.delete(TAG)
        if self.viewer.tool is self:
            self.viewer.tool = self.prev
        self._start = None

    def cancel(self):
        """Escape: abandon the rect. True when something was cancelled."""
        if not self.armed:
            return False
        self._disarm()
        if self.on_cancel is not None:
            self.on_cancel()
        return True

    # -- the canvas tool protocol ------------------------------------------ #
    def on_press(self, e):
        self._start = (e.x, e.y)
        return True

    def on_move(self, e):
        if self._start is None:
            return True
        c = self.viewer.canvas
        c.delete(TAG)
        x0, y0 = self._start
        c.create_rectangle(x0, y0, e.x, e.y, outline=COLOR, width=2,
                           dash=(6, 3), tags=("draw", TAG))
        return True

    def on_release(self, e):
        start = self._start
        self._disarm()
        if start is None:
            return
        a = self.viewer.screen_to_image(*start)
        b = self.viewer.screen_to_image(e.x, e.y)
        self.on_rect(min(a[0], b[0]), min(a[1], b[1]),
                     max(a[0], b[0]), max(a[1], b[1]))
