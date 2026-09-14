"""The prime's phase timings: recorded by the pipeline, rendered on the
engine's log line. `prime=34311ms` with visible MSC lines summing to two
seconds is what this exists to explain."""
import json

import numpy as np
import pytest

from msseg.mscoupon.engine import build_timings_brief


class _Pipe:
    def __init__(self, phases):
        self._phases = phases

    def build_timings(self):
        return dict(self._phases)


def test_brief_renders_phases_in_order_without_the_total():
    pipe = _Pipe([("value_range", 12.4), ("msc", 20311.0), ("stat_bank", 5000.6),
                  ("accumulate", 6001.2), ("total", 31325.2)])
    assert build_timings_brief(pipe) == " [value_range=12 msc=20311 stat_bank=5001 accumulate=6001]"


def test_brief_is_empty_on_an_extension_without_it():
    class Old:
        pass
    assert build_timings_brief(Old()) == ""
    assert build_timings_brief(_Pipe([("total", 3.0)])) == ""


def test_extension_records_the_phases_and_they_add_up():
    engine = pytest.importorskip("msseg.mscoupon")
    params = json.dumps({"statistics": {"channels": ["base", {"kind": "blur", "sigmas": [1.0]}],
                                        "reductions": ["mean", "std"]}})
    rng = np.random.default_rng(2)
    base = rng.random((40, 48), dtype=np.float32)
    pipe = engine.prime_slice(base, base, params)
    if not hasattr(pipe, "build_timings"):
        pytest.skip("extension predates build_timings")
    bt = pipe.build_timings()
    keys = list(bt)
    assert keys[-1] == "total"
    assert "msc" in keys and "accumulate" in keys and "select" in keys
    assert keys.index("msc") < keys.index("accumulate") < keys.index("select")
    assert all(v >= 0.0 for v in bt.values())
    phases = sum(v for k, v in bt.items() if k != "total")
    assert 0.5 * bt["total"] - 1.0 <= phases <= bt["total"] + 1.0
    brief = build_timings_brief(pipe)
    assert brief.startswith(" [value_range=") and "msc=" in brief and "total=" not in brief
