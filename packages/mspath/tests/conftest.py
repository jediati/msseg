"""Put the four source trees on sys.path so `pytest packages/mspath/tests` runs
from a source checkout without PYTHONPATH (an installed wheel set wins over
these when present, since they are appended)."""
import os
import sys

_PACKAGES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _rel in ("mspath/src", "mslabeler/src", "mscoupon/src", "msseg-viz/src"):
    _p = os.path.join(_PACKAGES, *_rel.split("/"))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)
