"""The stage strip: where the item on screen stands for the active task.

::

    [msc]--[stats]--+--[classified]
           [model]--+

Each box is COMPUTED from what the engines already hold -- the field and
measurement fingerprints, the record's identity, the prediction's commit, the
compatibility gate, the model's training stamps -- never set by the code that
changes them, so it cannot drift. The model is a branch, not a link in the
chain: it depends on the task's annotations and the statistics columns, not on
this item's MSC, so a place classified from a kept record whose pipeline was
released is a real state ("cached" msc), not an inconsistency.

States: ``ok`` (green), ``stale`` (burnt orange: something upstream moved),
``none`` (grey: not computed), ``cached`` (hollow green: kept, but what made it
is not live), ``busy`` (blue, spinning), ``error`` (red). A box whose upstream
is stale, busy or failed is shown stale too -- it will be recomputed. A click
on a box opens the tab that edits it.

The app answers for the two boxes only it can judge (``_stage_field``,
``_stage_measure``) and names its measurement (``_measure_key``); the model and
classified boxes are the framework's.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .. import bundle as model_bundle

ORDER = ("msc", "stats", "model", "classified")
LABELS = {"msc": "msc", "stats": "stats", "model": "model", "classified": "classified"}
TABS = {"msc": "Processing", "stats": "Features", "model": "Model", "classified": "Annotation"}
UPSTREAM = {"stats": ("msc",), "classified": ("stats", "model")}
# (row, col, feeds): the model joins the chain below it.
LAYOUT = {"msc": (0, 0, ("stats",)), "stats": (0, 1, ("classified",)),
          "model": (1, 1, ("classified",)), "classified": (0, 2, ())}
_BLOCKING = ("stale", "busy", "error")

State = Tuple[str, str, str]          # (state, tip, text)


def _norm(v) -> State:
    if v is None:
        return ("none", "", "")
    v = tuple(v)
    return (str(v[0]), str(v[1]) if len(v) > 1 else "", str(v[2]) if len(v) > 2 else "")


def propagate(states: Dict[str, tuple]) -> Dict[str, State]:
    """Downstream of a stale / busy / failed box, a box that reads ``ok`` is
    stale: what it shows will be recomputed. Pure, in dependency order."""
    out = {k: _norm(states.get(k)) for k in ORDER}
    for key in ORDER:
        state, tip, text = out[key]
        if state not in ("ok", "cached"):
            continue
        for up in UPSTREAM.get(key, ()):
            if out[up][0] in _BLOCKING:
                why = "is being recomputed" if out[up][0] == "busy" else "changed"
                out[key] = ("stale", f"{tip} -- but {up} {why}, so this is out of date"
                            if tip else f"{up} {why}, so this is out of date", text)
                break
    return out


def boxes(states: Dict[str, State]) -> List[dict]:
    """The canvas strip's box list (``SliceCanvas.set_stages``)."""
    out = []
    for key in ORDER:
        state, tip, text = states[key]
        row, col, feeds = LAYOUT[key]
        out.append({"key": key, "label": LABELS[key], "state": state, "tip": tip,
                    "text": text, "row": row, "col": col, "feeds": list(feeds)})
    return out


class StagesMixin:
    """The strip for ``AnnotationShell``. Refreshed from ``_update_busy`` (every
    engine event, preview and badge), navigation, task switches, store edits,
    model installs and predictions, with the hint poll as the backstop."""

    STAGE_STRIP = True

    # -- the app's answers ------------------------------------------------ #
    def _stage_field(self, key) -> Optional[tuple]:
        """(state, tip[, text]) of the item's MSC / field."""
        return ("none", "")

    def _stage_measure(self, key) -> Optional[tuple]:
        """(state, tip[, text]) of the item's statistics rows."""
        return ("none", "")

    def _measure_key(self) -> Optional[str]:
        """An opaque name for the panel's measurement (the base chain and the
        statistics); None when the app cannot say."""
        return None

    # -- the framework's boxes -------------------------------------------- #
    def _stage_model(self):
        if self._clf is None:
            return ("none", "No model yet: annotate, then Train (Annotation tab) "
                            "or load one.")
        names = list(self._clf_names or [])
        stack = self._task.model
        msg = self._stage_compat(names)
        if msg:
            return ("stale", f"{msg.splitlines()[0]} -- Train again or load a model "
                             "that matches this workflow.")
        mk = self._measure_key()
        if stack.trained_measure is not None and mk is not None and mk != stack.trained_measure:
            return ("stale", "The base channel or statistics changed since this model "
                             "was trained -- Train again (R).")
        if stack.trained_rev is not None and stack.trained_rev != self.store.rev:
            return ("stale", "Annotations or classes changed since this model was "
                             "trained -- Train again (R).")
        what = f"{self._clf_kind}, {len(names)} features"
        if stack.trained_rev is None:
            return ("ok", f"Loaded model ({what}).")
        return ("ok", f"Trained this session ({what}) on the current annotations.")

    def _stage_compat(self, names):
        """The compatibility gate's verdict for the loaded model, cached by
        what it depends on (the extension call is not free on a 0.7 s poll)."""
        sig = (self._measure_key(), id(self._clf), tuple(names),
               getattr(self, "_clf_scope", None), self._feature_scope())
        hit = getattr(self, "_stage_compat_cache", None)
        if hit is not None and hit[0] == sig:
            return hit[1]
        try:
            expected = self._expected_names_for(None)
        except Exception:
            expected = None
        msg = None
        if expected is not None:
            prof = "?"
            if 0 <= self.active_profile_idx < len(self.profiles):
                prof = self.profiles[self.active_profile_idx]["name"]
            msg = model_bundle.compat_message(names, expected, prof, "stage",
                                              getattr(self, "_clf_scope", None),
                                              self._feature_scope())
        self._stage_compat_cache = (sig, msg)
        return msg

    def _stage_classified(self, key):
        if self._clf is None:
            return ("none", "Nothing to classify with yet.")
        pr = self._pred.get(key)
        if pr is None:
            return ("none", "Not classified -- press C to classify this item.")
        rec = self.regions.record(key)
        if rec is None or pr[0] != rec.get("commit"):
            return ("stale", "Classified on an earlier version of these regions -- "
                             "press C.")
        ctx = getattr(self, "_clf_context", None)
        if (ctx is not None and getattr(ctx, "labels", None) is not None
                and getattr(self, "_pred_store_rev", None) != self.store.rev):
            return ("stale", "The annotations changed, and neighbours' labels are "
                             "inputs to this model -- press C.")
        return ("ok", "Classified with the current model.")

    # -- the strip -------------------------------------------------------- #
    def _stage_status(self):
        """The strip's boxes for the current item, or None (no item: browsing
        a row the task does not work, or nothing loaded)."""
        cur = self._current()
        if cur is None:
            return None
        key = self.catalogue.key_of(*cur)
        if key is None:
            return None
        states = {"msc": self._stage_field(key), "stats": self._stage_measure(key),
                  "model": self._stage_model(), "classified": self._stage_classified(key)}
        busy = getattr(self, "_stage_busy", None)
        if busy is not None and busy[0] in states:
            states[busy[0]] = ("busy", busy[1], busy[2] if len(busy) > 2 else "")
        return boxes(propagate(states))

    def _refresh_stages(self):
        v = getattr(self, "viewer", None)
        if v is None or not hasattr(v, "set_stages"):
            return
        try:
            v.set_stages(self._stage_status())
        except Exception as exc:              # a status display must never raise
            self._log(f"stage strip: {type(exc).__name__}: {exc}")

    def _set_stage_busy(self, stage, text="", detail=""):
        """Spin `stage` (None = nothing spins) and refresh."""
        self._stage_busy = None if stage is None else (stage, text, detail)
        self._refresh_stages()

    def _on_stage_click(self, key):
        tab = TABS.get(key)
        if tab:
            self._show_center_tab(tab)
