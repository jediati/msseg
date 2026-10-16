"""``ComputeEngine`` field slots: a primed stack, its records and its commit
belong to one field; ``use_field`` stashes the outgoing field and restores
another's, the least recently used field beyond ``MSSEG_COUPON_FIELDS`` is
evicted, and a removed row leaves every field in step with the sequences."""
import pytest

from msseg.mscoupon.engine import ComputeEngine, coupon_fields


class _Pipe:
    def __init__(self):
        self.released = 0

    def release_gpu(self):
        self.released += 1


def _stack(files):
    return [{"files": list(f), "base": [None] * len(f), "filtered": [None] * len(f),
             "pipes": [_Pipe() for _ in f], "normalizers": [None] * len(f),
             "color": [None] * len(f)} for f in files]


def _prime(eng, field, files, snapshot):
    """What a Run does: fill the active slot, name its field, commit."""
    eng.run_active = True
    eng._run_field = field
    eng.work_q.put(("done", _stack(files)))
    assert eng.poll() == [("primed",)]
    eng.commit_selection(snapshot)
    eng.slices[(0, 0)] = {"commit": eng.commit_id, "labels": field}
    return eng.commit_id


def test_switching_back_finds_the_stack_records_and_commit():
    eng = ComputeEngine(lambda *a: {})
    ca = _prime(eng, "X", [["a.tif", "b.tif"]], {"pct": 5})
    stack_a = eng.primed
    assert eng.active_field == "X" and eng.has_field("X")

    assert eng.use_field("Y", {"tag": "A"}) is None, "no stack in Y yet"
    assert eng.primed == [] and eng.record(0, 0) is None
    assert eng.has_field("X") and not eng.has_field("Y")
    cb = _prime(eng, "Y", [["a.tif", "b.tif"]], {"pct": 5})
    assert cb != ca, "the same snapshot on another field is another identity"

    assert eng.use_field("X", {"tag": "B"}) == {"tag": "A"}
    assert eng.primed is stack_a
    assert eng.commit_id == ca and eng.record(0, 0)["labels"] == "X"
    # The panel recommits the same snapshot: the same id, the same records.
    eng.commit_selection({"pct": 5})
    assert eng.commit_id == ca
    assert eng.use_field("Y") == {"tag": "B"} and eng.record(0, 0)["labels"] == "Y"
    assert sorted(eng.fields) == ["X", "Y"] and eng.fields[-1] == "Y"


def test_the_outgoing_stack_releases_its_gpu_residue():
    eng = ComputeEngine(lambda *a: {})
    _prime(eng, "X", [["a.tif"]], {})
    eng._gpu_active = (0, 0)
    pipe = eng.primed[0]["pipes"][0]
    eng.use_field("Y", {})
    assert pipe.released == 1 and eng._gpu_active is None


def test_least_recently_used_field_is_evicted(monkeypatch):
    monkeypatch.setenv("MSSEG_COUPON_FIELDS", "2")
    assert coupon_fields() == 2
    eng = ComputeEngine(lambda *a: {})
    _prime(eng, "X", [["a.tif"]], {})
    eng.use_field("Y", {})
    _prime(eng, "Y", [["a.tif"]], {})
    eng.use_field("X", {})                 # X is now the most recent
    eng.use_field("Z", {})                 # empty: two primed fields, still fits
    assert eng.has_field("X") and eng.has_field("Y")
    _prime(eng, "Z", [["a.tif"]], {})      # three primed: Y (least recent) goes
    assert eng.has_field("Z") and eng.has_field("X") and not eng.has_field("Y")
    assert eng.use_field("Y", {}) is None


def test_one_field_keeps_only_the_active_stack(monkeypatch):
    monkeypatch.setenv("MSSEG_COUPON_FIELDS", "1")
    eng = ComputeEngine(lambda *a: {})
    _prime(eng, "X", [["a.tif"]], {})
    eng.use_field("Y", {})
    assert eng.has_field("X"), "an empty active slot does not count"
    _prime(eng, "Y", [["a.tif"]], {})
    assert not eng.has_field("X")


@pytest.mark.parametrize("bad", ["", "zero", "0"])
def test_the_bound_falls_back_or_clamps(monkeypatch, bad):
    monkeypatch.setenv("MSSEG_COUPON_FIELDS", bad)
    assert coupon_fields() == (1 if bad == "0" else 2)


def test_a_run_on_an_edited_field_replaces_the_active_slot():
    """A Run fills the ACTIVE slot: an in-place chain edit + Run re-keys it
    (the old field is not kept beside it), as before field slots."""
    eng = ComputeEngine(lambda *a: {})
    _prime(eng, "X", [["a.tif"]], {})
    _prime(eng, "X2", [["a.tif"]], {})
    assert eng.fields == ["X2"]


def test_reset_drops_every_field_or_only_the_active_one():
    eng = ComputeEngine(lambda *a: {})
    _prime(eng, "X", [["a.tif"]], {})
    eng.use_field("Y", {})
    _prime(eng, "Y", [["a.tif"]], {})
    eng.reset(all_fields=False)
    assert eng.has_field("X") and not eng.has_field("Y")
    eng.reset()
    assert not eng.has_field("X") and eng.fields == []


def test_removed_rows_leave_stashed_fields_in_step():
    eng = ComputeEngine(lambda *a: {})
    _prime(eng, "X", [["a.tif", "b.tif"], ["c.tif"]], {})
    eng.slices[(1, 0)] = {"commit": eng.commit_id, "labels": "c"}
    eng.use_field("Y", {})
    _prime(eng, "Y", [["a.tif", "b.tif"], ["c.tif"]], {})
    # Drop slice b from sequence 0 everywhere, then sequence 0 itself.
    eng.drop_slice(0, 1)
    eng.drop_stashed(0, ["a.tif", "b.tif"], li=1)
    eng.drop_sequence(0)
    eng.drop_stashed(0, ["a.tif"])
    eng.use_field("X", {})
    assert [p["files"] for p in eng.primed] == [["c.tif"]]
    assert eng.record(0, 0)["labels"] == "c", "the record moved down with its sequence"


def test_a_stashed_stack_out_of_step_is_dropped():
    eng = ComputeEngine(lambda *a: {})
    _prime(eng, "X", [["a.tif"]], {})
    eng.use_field("Y", {})
    eng.drop_stashed(0, ["other.tif"])
    assert not eng.has_field("X")
    _prime(eng, "Y", [["a.tif"]], {})
    eng.use_field("X", {})
    assert eng.check_stashed([{"files": ["z.tif"]}]) == 1
    assert not eng.has_field("Y")


def test_no_switch_while_a_run_primes():
    eng = ComputeEngine(lambda *a: {})
    eng.run_active = True
    with pytest.raises(RuntimeError):
        eng.use_field("Y")


def test_an_assembly_in_flight_stays_on_its_stack():
    """The worker binds the stack it launched on; a switch supersedes its
    result (the token) and never hands it the other field's pipes."""
    eng = ComputeEngine(lambda *a: {})
    _prime(eng, "X", [["a.tif"]], {})
    stack_x = eng.primed
    eng.use_field("Y", {})
    _prime(eng, "Y", [["a.tif"]], {})
    seen = []
    eng._slice_result = lambda si, li, params, ext, np, tm, primed=None: (
        seen.append(primed), {"labels": None, "kept": set(), "n_feat": 0})[1]
    token = eng.asm_token
    eng._assemble_worker(token, 0, {"pct": 1, "queries": [], "pixels": [],
                                    "connectivity": 6, "level": "slice", "li": 0},
                         stack_x)
    assert seen == [stack_x]
    eng.use_field("X", {})                 # the token moves on a switch
    kind, (tok, _si, _out) = eng.work_q.get_nowait()
    assert kind == "assembly" and tok != eng.asm_token
