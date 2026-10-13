"""Foundation-model region features: a ViT's per-patch tokens pooled per region.

    spec     FoundationSpec / Basis, column names, schema entries (numpy only)
    extract  the tiled token grid (torch + timm, imported lazily)
    project  PCA basis fit + projection (numpy)
    pool     regions gather the cells they cover -> fm<hash8>__ columns (numpy)
    cache    TokenCache: grids by (item, level, halo, extract key), memory + disk

Phase 0 of docs/design_foundation_features.md: the offline experiment
(packages/mspath/experiments/fm_ablation.py) uses these; nothing in the
labeler reads them yet.
"""
from .spec import (Basis, FoundationSpec, MODELS, POOLS, column_names,  # noqa: F401
                   is_foundation_column, schema_entries)
