"""Item keys: what ``annotations.json`` stores, so they must round-trip exactly.

A key that parses back to a different Item silently reattaches somebody's
labels to a different place on a slide, which is the worst failure this package
can have and the cheapest to test against.
"""
import os

import pytest

from msseg.mspath import items as I


def test_overview_and_roi_keys():
    ov = I.overview("wsi/a.svs", 4)
    roi = I.roi("wsi/a.svs", 0, 12000, 4000, 4096, 4096)
    assert ov.key == "wsi/a.svs@4"
    assert roi.key == "wsi/a.svs@0#12000,4000,4096,4096"
    assert ov.is_overview and not roi.is_overview
    assert (ov.kind, roi.kind) == ("overview", "roi")


@pytest.mark.parametrize("item", [
    I.overview("wsi/a.svs", 0),
    I.overview("wsi/a.svs", 9),
    I.roi("wsi/a.svs", 0, 0, 0, 1, 1),
    I.roi("deep/folder/name.ome.tiff", 3, 12000, 4000, 4096, 2048),
    I.roi("with spaces/a b.svs", 2, 5, 7, 9, 11),
    I.roi("has@at/and#hash.svs", 1, 1, 2, 3, 4),      # the separators, in the slide name
])
def test_keys_round_trip(item):
    assert I.parse_key(item.key) == item


@pytest.mark.parametrize("bad", ["", "nonsense", "a.svs", "a.svs@", "a.svs@x",
                                 "a.svs@0#1,2,3", "a.svs@0#a,b,c,d", None])
def test_non_keys_parse_to_none(bad):
    assert I.parse_key(bad) is None


def test_a_level_change_is_a_different_item():
    """Not a nicety: the coarse and fine decompositions of the same tissue are
    not nested, so labels drawn at one level must not reattach to another."""
    a = I.roi("wsi/a.svs", 0, 100, 200, 64, 64)
    b = I.roi("wsi/a.svs", 1, 100, 200, 64, 64)
    assert a.key != b.key and I.parse_key(a.key) != I.parse_key(b.key)


def test_slide_id_is_folder_qualified_by_name():
    """Basenames collide across a session's folders, so the folder is part of
    the identity -- but by NAME, so a session survives the data moving."""
    a = I.slide_id(os.path.join("C:", "data", "batch1"), os.path.join("x", "s.svs"))
    b = I.slide_id(os.path.join("D:", "elsewhere", "batch1"), os.path.join("y", "s.svs"))
    assert a == b == "batch1/s.svs"
    assert I.slide_id(os.path.join("d", "batch2"), "s.svs") != a
    # a trailing separator must not eat the folder's name
    assert I.slide_id("batch1" + os.sep, "s.svs") == "batch1/s.svs"


def test_level_rect_is_the_rect_in_that_levels_pixels():
    roi = I.roi("s", 4, 1600, 800, 3200, 1600)
    assert roi.level_rect(16.0) == (100, 50, 200, 100)
    assert roi.level_rect(1.0) == (1600, 800, 3200, 1600)
    # a rect that would round to nothing still has one pixel
    assert I.roi("s", 9, 0, 0, 4, 4).level_rect(512.0)[2:] == (1, 1)
    with pytest.raises(ValueError):
        I.overview("s", 4).level_rect(16.0)


def test_labels_name_the_slide_and_the_place():
    assert "a.svs" in I.overview("wsi/a.svs", 4).label()
    assert "overview" in I.overview("wsi/a.svs", 4).label()
    text = I.roi("wsi/a.svs", 0, 12000, 4000, 4096, 2048).label()
    assert "4096x2048" in text and "12000" in text and "L0" in text
