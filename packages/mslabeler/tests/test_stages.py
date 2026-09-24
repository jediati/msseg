"""The stage strip's rule: downstream of a stale / busy / failed box, what
reads ok is out of date; the model is a branch into `classified`, never
downstream of this item's MSC."""
from msseg.labeler.panels.stages import LAYOUT, ORDER, TABS, boxes, propagate


def states(**kw):
    base = {k: ("ok", f"{k} fine") for k in ORDER}
    base.update(kw)
    return base


def test_all_ok_stays_ok():
    out = propagate(states())
    assert {k: v[0] for k, v in out.items()} == {k: "ok" for k in ORDER}


def test_a_stale_msc_makes_everything_downstream_of_it_stale_but_not_the_model():
    out = propagate(states(msc=("stale", "edited")))
    assert out["stats"][0] == "stale" and out["classified"][0] == "stale"
    assert out["model"][0] == "ok", "the model does not depend on this item's MSC"
    assert "msc changed" in out["stats"][1] and out["stats"][1].startswith("stats fine")


def test_a_busy_model_makes_classified_stale_with_a_reason():
    out = propagate(states(model=("busy", "Training", "3/10")))
    assert out["model"] == ("busy", "Training", "3/10")
    assert out["classified"][0] == "stale" and "being recomputed" in out["classified"][1]


def test_grey_upstream_does_not_invent_staleness_and_cached_counts_as_ok():
    out = propagate(states(msc=("cached", "released"), model=("none", "no model"),
                           classified=("none", "")))
    assert out["msc"][0] == "cached" and out["stats"][0] == "ok"
    assert out["classified"][0] == "none"
    out = propagate(states(msc=("error", "unreadable")))
    assert out["stats"][0] == "stale"


def test_boxes_carry_layout_and_tabs_cover_every_box():
    b = boxes(propagate(states(model=("busy", "Fitting", "1/2"))))
    assert [x["key"] for x in b] == list(ORDER)
    model = [x for x in b if x["key"] == "model"][0]
    assert (model["row"], model["col"], model["feeds"]) == (1, 1, ["classified"])
    assert model["text"] == "1/2" and model["state"] == "busy"
    assert set(TABS) == set(ORDER) == set(LAYOUT)
    assert propagate({"msc": None})["stats"] == ("none", "", "")
