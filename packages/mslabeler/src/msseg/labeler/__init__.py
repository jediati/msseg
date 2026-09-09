"""msseg.labeler -- the labeler framework shared by the MSSeg labelers.

Headless layers (annotation store, region-graph tools, classifier search,
model bundles) import without Tk; the shell, panels and canvas import tkinter
only when imported themselves. Import submodules explicitly::

    from msseg.labeler import labeling, magic_fill, model_search
    from msseg.labeler.canvas import SliceCanvas

Nothing heavy is imported here on purpose, so ``import msseg.labeler`` is safe
in a worker process or on a machine without tkinter or scikit-learn.
"""
__version__ = "0.1.0"

from .protocols import (ItemKey, ItemCatalogue, RegionProvider, RegionRecord,   # noqa: E402,F401
                        ImageSource, LabelLayer, FeatureTableLike)
