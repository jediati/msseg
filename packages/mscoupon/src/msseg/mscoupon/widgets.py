"""Small reusable Tk widgets shared by the mscoupon apps (viewer + labeler)."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .common import _bind_click_to_value, _wheel_delta


class ScrollFrame(ttk.Frame):
    """A vertically scrollable panel: a canvas carries the real content frame,
    exposed as ``.inner``, so callers pack sections into it exactly as they
    would into a plain frame.

    Wheel scrolling is armed only while the pointer is over the panel: the
    slice canvas binds the wheel to zoom (viewer_canvas.py), so a permanent
    bind_all would hijack it. Widgets with their own wheel binding (e.g. a
    scrolled listbox) return "break" before the panel handler fires.
    """

    def __init__(self, parent, width=376, canvas_width=360):
        super().__init__(parent, width=width)
        self.canvas = tk.Canvas(self, width=canvas_width, highlightthickness=0,
                                borderwidth=0, takefocus=0)
        scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = ttk.Frame(self.canvas)
        self._window = self.canvas.create_window((0, 0), window=self.inner,
                                                 anchor="nw")
        self.inner.bind("<Configure>", self._on_content_resize)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)

    def _on_content_resize(self, _event=None):
        """Sections were added/removed -> refresh the scrollable extent."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_resize(self, event):
        """Hold the inner frame at the canvas width so `fill="x"` still spans."""
        self.canvas.itemconfigure(self._window, width=event.width)

    def _scrollable(self):
        box = self.canvas.bbox("all")
        return bool(box) and box[3] > self.canvas.winfo_height()

    def _bind_wheel(self, _event=None):
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.bind_all(seq, self._on_wheel)

    def _unbind_wheel(self, _event=None):
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.unbind_all(seq)

    def _on_wheel(self, event):
        if self._scrollable():
            self.canvas.yview_scroll(_wheel_delta(event), "units")


def jump_scale(parent, **kw):
    """A ttk.Scale that jumps to the click/drag position (see _bind_click_to_value)."""
    s = ttk.Scale(parent, **kw)
    _bind_click_to_value(s)
    return s


def scrolled_listbox(parent, **kw):
    """A Listbox with a vertical scrollbar + mouse-wheel scrolling (production
    folders have thousands of files)."""
    frame = ttk.Frame(parent)
    frame.pack(fill="x", padx=4, pady=2)
    lb = tk.Listbox(frame, **kw)
    sb = ttk.Scrollbar(frame, orient="vertical", command=lb.yview)
    lb.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y")
    lb.pack(side="left", fill="both", expand=True)
    # "break" keeps the wheel here instead of also scrolling the left panel.
    for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):   # Button-4/5 = X11
        lb.bind(seq, lambda e, w=lb: (w.yview_scroll(_wheel_delta(e), "units"), "break")[1])
    return lb
