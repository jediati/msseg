"""Region embeddings: one vector per region, chosen, resolved and cached in
one place.

Many tools read "a vector per region" and compare two of them: the magic
fill and the blobber, the seam livewire's table tolls, the edge model, the
seam model's pair block, the latent ring head, a polyline task's embedding
input and the Inspect panel. This module says WHICH vectors (an
``EmbeddingRef``), what is on offer right now (``options``), computes them
for an item (``resolve``), caches them by what they are a function of
(``EmbeddingCache``) and compares them (``distance`` / ``similarity``).
Design: ``docs/design_region_embeddings.md``.

Sources:

  columns   the statistics table itself. ``block`` picks the columns:
            ``row`` (every column except positions and histogram bins --
            the magic fill's ``cosine`` row), ``channels`` (the ``mean_<c>``
            columns of ``channels``; the ``mean`` metric), ``gauss`` (means
            and stds; the ``bhattacharyya`` metric), ``histogram`` (the bin
            fractions), ``family`` (one column family -- ``fm<hash>``,
            ``emb<hash>``, a context kind, ``hist``; see ``families``) or
            ``names`` (explicit). ``norm`` = ``slice`` z-scores each column
            over the item (the magic fill's convention), ``none`` reads raw.
  net       a hidden layer of a region net (the active task's, or another
            task's by ``provider`` = its uid).
  head      a hidden layer of the latent-ring head's net.
  proba     the classifier's class probabilities.

**One layer convention**: ``layer`` 0 is what the net's last step sees
(FeatureSubset + scaler applied: the model's own standardised input), ``k >=
1`` the hidden layer ``h<k>``, ``None`` the LAST hidden layer whatever the
depth. A layer the net does not have is refused with a reason, never read
from another layer. ``legacy_layer`` maps the older spellings (a Python
index into the hidden list, ``-1`` = last).

A slice-normalised vector (``columns`` + ``norm="slice"``) is a different
vector for the same region on another item, so ``Embedding.cross_item`` is
False for it: comparing across items needs a net layer, raw columns,
histograms, foundation / encoder families or probabilities.

Pure numpy (``np`` passed in, like ``magic_fill``). No Tk, no engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields as dc_fields
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import edge_model, magic_fill
from .fields import DEFAULT, FieldConventions
from .record_keys import RecordCache

SOURCES = ("columns", "net", "head", "proba")
BLOCKS = ("row", "channels", "gauss", "histogram", "family", "names")
NORMS = ("slice", "none")
# Distances: a non-negative dissimilarity (what a flood or a toll needs).
DISTANCES = ("euclidean", "cosine", "l2n", "bhattacharyya", "hellinger", "tv")
# The distances a flood over an embedding offers (the magic fill's picker).
FLOOD_DISTANCES = ("cosine", "euclidean", "l2n")
# Similarities: what the Inspect colouring reads (larger = more alike).
SIMILARITIES = ("cosine", "dot", "euclidean")
REDUCES = ("max", "mean")

# The family of a column that belongs to none of the named ones.
STATS_FAMILY = "stats"
HIST_FAMILY = "hist"
SEP = "__"


# --------------------------------------------------------------------------- #
# The reference
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EmbeddingRef:
    source: str = "columns"
    block: str = "row"                     # columns only
    family: str = ""                       # block == "family"
    channels: Tuple[str, ...] = ()         # block in channels / gauss / histogram ('' = all)
    names: Tuple[str, ...] = ()            # block == "names"
    layer: Optional[int] = None            # net / head: 0 = input, k = h<k>, None = last
    norm: str = "slice"                    # columns family / names
    provider: Optional[str] = None         # net: a task uid; None = the active task

    def __post_init__(self):
        if self.source not in SOURCES:
            raise ValueError(f"unknown embedding source {self.source!r}")
        if self.block not in BLOCKS:
            raise ValueError(f"unknown column block {self.block!r}")
        if self.norm not in NORMS:
            raise ValueError(f"unknown normalisation {self.norm!r}")

    # -- serialisation: only what differs from the default ---------------- #
    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for f in dc_fields(self):
            v = getattr(self, f.name)
            if v != f.default:
                out[f.name] = list(v) if isinstance(v, tuple) else v
        return out

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "EmbeddingRef":
        d = dict(d or {})
        known = {f.name for f in dc_fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        for k in ("channels", "names"):
            if k in d:
                d[k] = tuple(str(x) for x in (d[k] or ()))
        if "layer" in d:
            d["layer"] = None if d["layer"] is None else int(d["layer"])
        return cls(**d)

    def key(self) -> tuple:
        return (self.source, self.block, self.family, self.channels, self.names,
                self.layer, self.norm, self.provider)

    @property
    def cross_item(self) -> bool:
        """Is a region's vector the same function of the region on every item?"""
        return not (self.source == "columns" and self.norm == "slice"
                    and self.block in ("row", "channels", "family", "names"))

    def uses_model(self) -> bool:
        return self.source in ("net", "head", "proba")


def legacy_layer(index: int, depth: int) -> Optional[int]:
    """The older spelling -- a Python index into the HIDDEN layers (``-1`` =
    last, ``-2`` = previous, ``0`` = the first hidden layer) -- in this
    module's convention. ``-1`` maps to None (the last, whatever the depth);
    a negative index past the first layer raises ValueError."""
    i = int(index)
    if i == -1:
        return None
    if i < 0:
        k = int(depth) + i + 1
        if k < 1:
            raise ValueError(f"layer {i} was picked, but the net has {depth} hidden "
                             f"layer{'s' if depth != 1 else ''} -- pick another")
        return k
    return i + 1


def to_legacy_layer(layer: Optional[int]) -> int:
    """The inverse of ``legacy_layer`` for a hidden layer: None -> -1, k -> k-1."""
    if layer is None:
        return -1
    if int(layer) < 1:
        raise ValueError("the input layer has no legacy spelling")
    return int(layer) - 1


def resolve_layer(layer: Optional[int], depth: int) -> int:
    """The index into ``edge_model.embed_layers``'s output for `layer` on a
    net with `depth` hidden layers; ValueError (with the reason) when the
    net has no such layer."""
    depth = int(depth)
    if layer is None:
        if depth < 1:
            raise ValueError("no hidden layer to embed with")
        return depth
    k = int(layer)
    if 0 <= k <= depth:
        return k
    return _no_layer(k, depth)


def _no_layer(k, depth):
    raise ValueError(f"layer h{k} was picked, but the net has {depth} hidden "
                     f"layer{'s' if depth != 1 else ''} -- pick another")


def layer_name(k: int) -> str:
    return "input" if k == 0 else f"h{k}"


# --------------------------------------------------------------------------- #
# Model introspection
# --------------------------------------------------------------------------- #
def layers_of(pipeline) -> List[Tuple[int, Optional[int], str]]:
    """``[(k, width, name), ...]`` the pipeline can embed at: ``(0, n_in,
    "input")`` when the last step has earlier steps (a scaler) or hidden
    layers, then ``(k, width, "h<k>")`` per hidden layer. [] for anything
    else (no pipeline, or a bare estimator with nothing to read)."""
    steps = getattr(pipeline, "steps", None)
    if not steps:
        return []
    try:
        _net, hidden, _out = edge_model._hidden_layers(pipeline)
    except (TypeError, AttributeError):
        hidden = []
    out: List[Tuple[int, Optional[int], str]] = []
    if hidden:
        out.append((0, int(hidden[0][0].shape[0]), "input"))
    elif len(steps) > 1:
        n_in = getattr(steps[-1][1], "n_features_in_", None)
        out.append((0, None if n_in is None else int(n_in), "input"))
    for k, (W, _b) in enumerate(hidden, start=1):
        out.append((k, int(W.shape[1]), f"h{k}"))
    return out


def depth_of(pipeline) -> int:
    """How many hidden layers the pipeline has (0 for none / a forest)."""
    return sum(1 for k, _w, _n in layers_of(pipeline) if k >= 1)


# --------------------------------------------------------------------------- #
# Column families
# --------------------------------------------------------------------------- #
def family_of(name: str, conv: FieldConventions = DEFAULT) -> Optional[str]:
    """The family a column belongs to, or None for a positional column:
    ``hist`` for a histogram bin, the ``fm<hash8>`` / ``emb<hash8>`` prefix
    for foundation / encoder columns, the context kind (``ring_mean[area]``,
    ``nbr_class``) for a context column, else ``stats``."""
    if name in conv.positional or name == conv.id_field:
        return None
    if conv.is_histogram_column(name):
        return HIST_FAMILY
    from . import context, embedding
    from .foundation import spec as fm_spec
    if context.is_context_column(name):
        return name.partition(SEP)[0]
    if fm_spec.is_foundation_column(name) or embedding.is_embedding_column(name):
        return name.partition(SEP)[0]
    return STATS_FAMILY


def families(names: Sequence[str], conv: FieldConventions = DEFAULT) -> Dict[str, List[str]]:
    """family -> its columns, in table order (families in first-seen order)."""
    out: Dict[str, List[str]] = {}
    for n in names:
        f = family_of(n, conv)
        if f is not None:
            out.setdefault(f, []).append(n)
    return out


def row_names(names: Sequence[str], conv: FieldConventions = DEFAULT) -> List[str]:
    """The ``row`` block: every column except positions and histogram bins --
    exactly what the magic fill's ``cosine`` reads."""
    return [n for n in names if n not in conv.positional and not conv.is_histogram_column(n)]


def family_label(fam: str) -> str:
    if fam == STATS_FAMILY:
        return "statistics"
    if fam == HIST_FAMILY:
        return "histograms"
    if fam.startswith("fm"):
        return f"foundation {fam}"
    if fam.startswith("emb"):
        return f"encoder {fam}"
    return f"context {fam}"


# --------------------------------------------------------------------------- #
# Options: what can be embedded right now
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EmbeddingOption:
    ref: EmbeddingRef
    label: str
    dim: Optional[int]
    group: str                      # statistics | learned columns | model | other tasks
    why_not: Optional[str] = None

    @property
    def usable(self) -> bool:
        return self.why_not is None


def _missing(want: Sequence[str], have: Sequence[str]) -> Optional[str]:
    if have is None:
        return None
    hs = set(have)
    miss = [c for c in want if c not in hs]
    if not miss:
        return None
    more = f" (+{len(miss) - 3} more)" if len(miss) > 3 else ""
    return f"needs {', '.join(miss[:3])}{more}, which this table lacks"


def net_options(pipeline, *, source="net", provider=None, owner="region net",
                input_names: Optional[Sequence[str]] = None,
                have: Optional[Sequence[str]] = None,
                why_not: Optional[str] = None) -> List[EmbeddingOption]:
    """One option per layer of `pipeline` (``layers_of``), labelled
    ``"h2 · 16-d (region net)"``; `why_not` greys them all (a provider's
    gate), else missing input columns do."""
    layers = layers_of(pipeline)
    if not layers:
        return []
    why = why_not or (None if input_names is None else _missing(input_names, have))
    last = max(k for k, _w, _n in layers)
    out = []
    for k, w, name in layers:
        dim = f" · {w}-d" if w is not None else ""
        is_last = k == last and k >= 1
        tail = ", last" if is_last else (", scaled input" if k == 0 else "")
        # The last hidden layer is offered as "last" (layer None): what the
        # edge model reads by default, and still the last after a retrain
        # changes the depth.
        out.append(EmbeddingOption(
            EmbeddingRef(source=source, layer=None if is_last else k, provider=provider),
            f"{name}{dim} ({owner}{tail})", w,
            "model" if provider is None else "other tasks", why))
    return out


def options(table_names: Sequence[str], *, stack=None, pred_ok: bool = False,
            providers: Sequence[Tuple[str, str, Any, Optional[str]]] = (),
            conv: FieldConventions = DEFAULT) -> List[EmbeddingOption]:
    """Every embedding on offer for a table with columns `table_names`:

    * statistics -- ``row``, the channel means, histograms, and a block per
      column family that is not plain statistics (foundation, encoder,
      context kinds);
    * the active model (`stack`: ``.clf``, ``.names``, ``.latent``) -- its
      input and every hidden layer, and the latent head's layers;
    * ``proba`` when `pred_ok` (a prediction is current for the item);
    * other tasks -- `providers` as ``(uid, name, stack, why_not)``.

    An option that cannot be resolved here carries ``why_not``."""
    names = list(table_names)
    out: List[EmbeddingOption] = []
    rows = row_names(names, conv)
    if rows:
        out.append(EmbeddingOption(EmbeddingRef(), f"row · {len(rows)}-d (statistics, per item)",
                                   len(rows), "statistics"))
    chans = [n[len(conv.mean_prefix):] for n in names if n.startswith(conv.mean_prefix)]
    if chans:
        out.append(EmbeddingOption(EmbeddingRef(block="channels", channels=tuple(chans)),
                                   f"channel means · {len(chans)}-d (per item)", len(chans),
                                   "statistics"))
    fams = families(names, conv)
    if HIST_FAMILY in fams:
        out.append(EmbeddingOption(EmbeddingRef(block="histogram"),
                                   f"histograms · {len(fams[HIST_FAMILY])}-d", len(fams[HIST_FAMILY]),
                                   "statistics"))
    for fam, cols in fams.items():
        if fam in (STATS_FAMILY, HIST_FAMILY):
            continue
        norm = "none" if fam.startswith(("fm", "emb")) else "slice"
        out.append(EmbeddingOption(EmbeddingRef(block="family", family=fam, norm=norm),
                                   f"{family_label(fam)} · {len(cols)}-d", len(cols),
                                   "learned columns"))
    if stack is not None and getattr(stack, "clf", None) is not None:
        out += net_options(stack.clf, input_names=list(stack.names or ()), have=names)
        latent = getattr(stack, "latent", None)
        if latent is not None:
            out += net_options(latent.head, source="head", owner="latent head")
    if pred_ok:
        out.append(EmbeddingOption(EmbeddingRef(source="proba"), "class probabilities",
                                   None, "model"))
    for uid, pname, pstack, why in providers:
        clf = getattr(pstack, "clf", None)
        if clf is None:
            continue
        out += net_options(clf, provider=uid, owner=pname, why_not=why,
                           input_names=list(getattr(pstack, "names", None) or ()), have=names)
    return out


def short_label(ref: EmbeddingRef) -> str:
    """A label for `ref` that needs no model in hand (a restored view shows
    it before the model loads): ``"net h2"``, ``"net last @t_ab12cd"``..."""
    if ref.source in ("net", "head"):
        text = f"{ref.source} " + ("last" if ref.layer is None else layer_name(ref.layer))
        return text + (f" @{ref.provider}" if ref.provider else "")
    if ref.source == "proba":
        return "class probabilities"
    if ref.block == "family":
        return family_label(ref.family)
    return {"row": "row", "channels": "channel means", "gauss": "means + stds",
            "histogram": "histograms", "names": "named columns"}[ref.block]


def option_for(ref: EmbeddingRef, opts: Sequence[EmbeddingOption]) -> Optional[EmbeddingOption]:
    """The option whose ref matches `ref` (channel lists compared as sets of
    what is offered), or None."""
    k = ref.key()
    for o in opts:
        if o.ref.key() == k:
            return o
    return None


# --------------------------------------------------------------------------- #
# Resolving: vectors for an item
# --------------------------------------------------------------------------- #
@dataclass
class Embedding:
    Z: Any                          # float32[n_rows, d], row-aligned with the table
    ids: Any                        # the table's region ids (intp[n_rows])
    names: List[str]                # one per column of Z
    distance: str                   # the natural dissimilarity
    cross_item: bool
    ref: EmbeddingRef
    signature: Any = None
    rows_of: Dict[int, int] = field(default_factory=dict, repr=False)

    @property
    def dim(self) -> int:
        return int(self.Z.shape[1])

    def row_of(self, region_id: int) -> Optional[int]:
        if not self.rows_of:
            self.rows_of = {int(r): i for i, r in enumerate(self.ids.tolist())}
        return self.rows_of.get(int(region_id))

    def by_id(self, values, np, K: Optional[int] = None, fill=0.0):
        """Scatter a per-row array onto region ids: ``(out[K], mask[K])``."""
        ids = np.asarray(self.ids, np.intp)
        K = int(K if K is not None else (ids.max() + 1 if len(ids) else 0))
        out = np.full(K, fill, np.float32)
        mask = np.zeros(K, bool)
        ok = (ids >= 0) & (ids < K)
        out[ids[ok]] = np.asarray(values, np.float32)[ok]
        mask[ids[ok]] = True
        return out, mask


def _ids(table, conv, np):
    col = table.column(conv.id_field)
    if col is None:
        raise ValueError(f"feature table has no {conv.id_field} column")
    return np.asarray(col, np.intp)


def _zscore(X, np):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (X - mu) / sd


def _columns(table, ref, np, conv):
    if ref.block == "row":
        X, kind = magic_fill.row_vectors(table, "cosine", [], np, conv=conv)
        return X, kind, row_names(table.names, conv)
    if ref.block in ("channels", "gauss", "histogram"):
        avail = magic_fill.channel_names(table, conv)
        chans = [c for c in ref.channels if c in avail] if ref.channels else list(avail)
        if ref.block == "histogram":
            X, kind = magic_fill.row_vectors(table, "histogram", list(ref.channels), np, conv=conv)
            hist = conv.histogram_columns(table)
            picked = [c for c in ref.channels if c in hist] or list(hist)
            return X, kind, [n for c in picked for n in hist[c]]
        if not chans:
            raise ValueError("none of the picked channels is in the feature table")
        metric = "mean" if ref.block == "channels" else "bhattacharyya"
        X, kind = magic_fill.row_vectors(table, metric, chans, np, conv=conv)
        names = [conv.mean_of(c) for c in chans]
        if ref.block == "gauss":
            names += [conv.std_of(c) for c in chans]
        return X, kind, names
    if ref.block == "family":
        cols = families(table.names, conv).get(ref.family)
        if not cols:
            raise ValueError(f"the table has no {family_label(ref.family)} columns")
    else:
        cols = list(ref.names)
        miss = _missing(cols, table.names)
        if miss or not cols:
            raise ValueError(miss or "no columns named")
    from .training import TrainingSetBuilder
    X = TrainingSetBuilder.feature_matrix(table, cols, np)
    if ref.norm == "slice":
        X = _zscore(X, np)
    return X, "cosine", list(cols)


def _net_matrix(stack, table, np):
    from .training import TrainingSetBuilder
    names = list(getattr(stack, "names", None) or [])
    mat = TrainingSetBuilder.feature_matrix(table, names, np)
    if mat is None:
        raise ValueError(_missing(names, table.names) or "the table lacks the model's columns")
    return mat


def resolve(ref: EmbeddingRef, *, table, np, stack=None, pred_entry=None,
            head_input: Optional[Callable[[Any], Any]] = None,
            conv: FieldConventions = DEFAULT, signature=None) -> Embedding:
    """The vectors `ref` names for every row of `table`.

    `table` is the item's statistics table -- for ``net`` it must carry the
    model's columns, context columns included (the app passes its context
    table, as ``_predict_slice`` does). `stack` is the model (``.clf``,
    ``.names``, ``.latent``) for ``net`` / ``head`` -- the PROVIDER's stack
    for a ref with a provider. `pred_entry` is the item's prediction cache
    entry ``(commit, final, proba[K, C], ...)`` for ``proba``. `head_input`
    maps the base net's input matrix to the latent head's (``X ++ latent``;
    the app builds it over the item's graph). ValueError with the reason when
    the vectors cannot be built -- never a silent substitute."""
    ids = _ids(table, conv, np)
    if ref.source == "columns":
        X, kind, names = _columns(table, ref, np, conv)
        Z = np.asarray(X, np.float32)
    elif ref.source == "proba":
        if pred_entry is None or len(pred_entry) < 3:
            raise ValueError("class probabilities need a prediction at this commit - Classify first")
        X, kind = magic_fill.row_vectors(table, "proba", [], np, extra={"proba": pred_entry[2]},
                                         conv=conv)
        Z = np.asarray(X, np.float32)
        names = [f"p{k}" for k in range(Z.shape[1])]
    else:
        if stack is None or getattr(stack, "clf", None) is None:
            raise ValueError("no trained region model to embed with")
        mat = _net_matrix(stack, table, np)
        net = stack.clf
        if ref.source == "head":
            latent = getattr(stack, "latent", None)
            if latent is None:
                raise ValueError("the model has no latent head")
            if head_input is None:
                raise ValueError("the latent head's input needs the item's graph")
            mat = head_input(mat)
            net = latent.head
        k = resolve_layer(ref.layer, depth_of(net))
        Z = edge_model.embed_layers(net, mat, upto=k)[k]
        kind = "cosine"
        names = [f"{layer_name(k)}_{j:02d}" for j in range(Z.shape[1])]
    Z = np.nan_to_num(np.asarray(Z, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return Embedding(Z=Z, ids=ids, names=names, distance=kind, cross_item=ref.cross_item,
                     ref=ref, signature=signature)


def signature_of(ref: EmbeddingRef, commit, stack=None, store_rev=None) -> tuple:
    """What `ref`'s vectors on an item at `commit` are a function of: the
    ref, and for a model source the net's own weights (``net_hash`` -- a
    reloaded identical pickle still hits), its input names, its context spec
    and, when the net reads neighbour labels, the store revision. A
    ``proba`` ref is keyed by the prediction entry the caller holds."""
    sig = [commit, ref.key()]
    if ref.source == "columns" and ref.block in ("family", "names"):
        # These may read context columns, and a labels block moves with the store.
        sig.append(store_rev)
    if ref.source in ("net", "head") and stack is not None and getattr(stack, "clf", None) is not None:
        try:
            sig.append(edge_model.net_hash(stack.clf))
        except (TypeError, AttributeError):
            sig.append(id(stack.clf))
        sig.append(edge_model.names_hash(stack.names or ()))
        ctx = getattr(stack, "context", None)
        if ctx is not None and not ctx.empty():
            sig.append(ctx.key())
            if getattr(ctx, "labels", None) is not None:
                sig.append(store_rev)
        if ref.source == "head":
            sig.append(id(getattr(stack, "latent", None)))
    return tuple(sig)


class EmbeddingCache:
    """Per item: signature -> ``Embedding``, a few per item (least recently
    used out). The first consumer of an item's vectors pays for them."""

    def __init__(self, cap: Optional[int] = None):
        self._c = RecordCache(cap or 6)
        self.hits = 0
        self.misses = 0

    def get(self, item, signature) -> Optional[Embedding]:
        got = self._c.get(item, signature)
        if got is not None:
            self.hits += 1
        return got

    def put(self, item, signature, emb: Embedding) -> Embedding:
        emb.signature = signature
        return self._c.put(item, signature, emb)

    def resolve(self, item, ref: EmbeddingRef, *, commit, table, np, stack=None,
                store_rev=None, **kw) -> Embedding:
        """``resolve`` through the cache (``proba`` is never cached: it is a
        lookup, and its entry changes under the same commit)."""
        if ref.source == "proba":
            return resolve(ref, table=table, np=np, stack=stack, **kw)
        sig = signature_of(ref, commit, stack, store_rev)
        got = self.get(item, sig)
        if got is not None:
            return got
        self.misses += 1
        return self.put(item, sig, resolve(ref, table=table, np=np, stack=stack,
                                           signature=sig, **kw))

    def clear(self):
        self._c.clear()


# --------------------------------------------------------------------------- #
# Comparing
# --------------------------------------------------------------------------- #
def distance(X, I, J, kind, np):
    """dist(X[I], X[J]) elementwise: ``magic_fill.pairwise``'s kinds plus
    ``l2n`` (euclidean between L2-normalised rows, in [0, 2])."""
    if kind == "l2n":
        A = X[np.asarray(I, dtype=np.intp)]
        B = X[np.asarray(J, dtype=np.intp)]
        na = np.sqrt((A * A).sum(axis=-1, keepdims=True))
        nb = np.sqrt((B * B).sum(axis=-1, keepdims=True))
        A = A / np.where(na > 1e-12, na, 1.0)
        B = B / np.where(nb > 1e-12, nb, 1.0)
        return np.sqrt(((A - B) ** 2).sum(axis=-1)).astype(np.float64)
    return magic_fill.pairwise(X, I, J, kind, np)


def similarity(Z, anchors, kind, np, reduce: str = "max"):
    """``float32[n]``: how alike every row of `Z` is to the anchor vectors
    (``(m, d)`` or ``(d,)``), reduced over the anchors by ``max`` / ``mean``.
    ``cosine`` in [-1, 1], ``dot`` unbounded, ``euclidean`` as minus the
    distance (so larger is always more alike)."""
    if kind not in SIMILARITIES:
        raise ValueError(f"unknown similarity {kind!r}")
    if reduce not in REDUCES:
        raise ValueError(f"unknown reduction {reduce!r}")
    Z = np.asarray(Z, np.float32)
    A = np.atleast_2d(np.asarray(anchors, np.float32))
    if A.shape[0] == 0:
        return np.zeros(len(Z), np.float32)
    if A.shape[1] != Z.shape[1]:
        raise ValueError(f"anchor is {A.shape[1]}-d, the embedding {Z.shape[1]}-d")
    if kind == "cosine":
        nz = np.sqrt((Z * Z).sum(axis=1, keepdims=True))
        na = np.sqrt((A * A).sum(axis=1, keepdims=True))
        S = (Z / np.where(nz > 1e-12, nz, 1.0)) @ (A / np.where(na > 1e-12, na, 1.0)).T
    elif kind == "dot":
        S = Z @ A.T
    else:
        d2 = ((Z * Z).sum(1)[:, None] - 2.0 * (Z @ A.T) + (A * A).sum(1)[None, :])
        S = -np.sqrt(np.maximum(d2, 0.0))
    S = S.max(axis=1) if reduce == "max" else S.mean(axis=1)
    return np.asarray(S, np.float32)


def to_unit(sim, kind, np, mask=None):
    """Map a similarity onto [0, 1] for the colour ramp: cosine by
    ``(s + 1) / 2``, the unbounded kinds by min-max over `mask`'s rows."""
    s = np.asarray(sim, np.float32)
    if kind == "cosine":
        return np.clip((s + 1.0) * 0.5, 0.0, 1.0)
    sel = s if mask is None else s[np.asarray(mask, bool)]
    if not len(sel):
        return np.zeros_like(s)
    lo, hi = float(sel.min()), float(sel.max())
    if hi - lo < 1e-12:
        return np.ones_like(s)
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0)


def pair_block(Z, a, b, np, ops: Sequence[str] = ("absdiff", "prod")):
    """``float32[n, d * len(ops)]`` symmetric in (a, b): ``|Z_a - Z_b|`` and
    ``Z_a * Z_b`` in `ops` order (the edge and seam models' pair terms)."""
    a = np.asarray(a, np.intp)
    b = np.asarray(b, np.intp)
    za = np.asarray(Z, np.float32)[a]
    zb = np.asarray(Z, np.float32)[b]
    parts = []
    for op in ops:
        if op == "absdiff":
            parts.append(np.abs(za - zb))
        elif op == "prod":
            parts.append(za * zb)
        else:
            raise ValueError(f"unknown pair op {op!r}")
    if not parts:
        return np.zeros((len(a), 0), np.float32)
    return np.concatenate(parts, axis=1).astype(np.float32)


def top_k(Z, anchors, kind, np, k: int = 50, reduce: str = "max", exclude=()):
    """``(rows, sims)`` of the `k` rows most like the anchors, best first;
    rows in `exclude` (the anchors themselves) are skipped."""
    s = similarity(Z, anchors, kind, np, reduce)
    if len(exclude):
        s = s.copy()
        s[np.asarray(list(exclude), np.intp)] = -np.inf
    k = int(min(max(k, 0), len(s)))
    if k == 0:
        return np.zeros(0, np.intp), np.zeros(0, np.float32)
    part = np.argpartition(-s, k - 1)[:k]
    order = part[np.argsort(-s[part], kind="stable")]
    keep = np.isfinite(s[order])
    return order[keep], s[order][keep]


# --------------------------------------------------------------------------- #
# Looking at an embedding
# --------------------------------------------------------------------------- #
def project_2d(Z, np, max_fit: int = 20000, seed: int = 0):
    """``float32[n, 2]``: the rows on their first two principal axes, fitted
    on at most `max_fit` rows (seeded) and applied to all. The axes' signs
    are fixed (largest |loading| positive) so a refit does not mirror the
    picture."""
    Z = np.asarray(Z, np.float64)
    n, d = Z.shape if Z.ndim == 2 else (0, 0)
    if n == 0:
        return np.zeros((0, 2), np.float32)
    rows = np.arange(n)
    if n > max_fit:
        rows = np.sort(np.random.default_rng(seed).choice(n, max_fit, replace=False))
    mu = Z[rows].mean(axis=0)
    C = Z[rows] - mu
    if d == 0:
        return np.zeros((n, 2), np.float32)
    _u, _s, Vt = np.linalg.svd(C, full_matrices=False)
    V = Vt[:2].T
    if V.shape[1] < 2:
        V = np.concatenate([V, np.zeros((d, 2 - V.shape[1]))], axis=1)
    for j in range(V.shape[1]):
        if V[np.argmax(np.abs(V[:, j])), j] < 0:
            V[:, j] = -V[:, j]
    return ((Z - mu) @ V).astype(np.float32)


def separability(Z, cls, groups, np, k: int = 5, max_rows: int = 20000, seed: int = 0):
    """How well an embedding separates the annotated classes, scored leave-
    one-group-out (a group = an item): ``{"knn", "centroid", "silhouette",
    "n", "groups"}``. `cls` is per row (0 = unlabelled, ignored); kNN and
    nearest-centroid run in cosine geometry on L2-normalised rows. With one
    group, a seeded half split stands in for the groups. None when fewer
    than two classes are labelled."""
    Z = np.asarray(Z, np.float64)
    cls = np.asarray(cls, int)
    groups = np.asarray(groups)
    lab = np.flatnonzero(cls > 0)
    if len(np.unique(cls[lab])) < 2:
        return None
    rng = np.random.default_rng(seed)
    if len(lab) > max_rows:
        lab = np.sort(rng.choice(lab, max_rows, replace=False))
    X = Z[lab]
    nx = np.sqrt((X * X).sum(1, keepdims=True))
    X = X / np.where(nx > 1e-12, nx, 1.0)
    y = cls[lab]
    g = groups[lab]
    ug = np.unique(g)
    if len(ug) < 2:
        g = (rng.random(len(lab)) < 0.5).astype(int)
        ug = np.unique(g)
    knn_hit = cen_hit = total = 0
    for gv in ug:
        te = g == gv
        tr = ~te
        if not te.any() or len(np.unique(y[tr])) < 2:
            continue
        S = X[te] @ X[tr].T
        kk = min(k, int(tr.sum()))
        nn = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        ytr = y[tr]
        votes = ytr[nn]
        classes = np.unique(ytr)
        counts = np.stack([(votes == c).sum(1) for c in classes], axis=1)
        knn_hit += int((classes[counts.argmax(1)] == y[te]).sum())
        cents = np.stack([X[tr][ytr == c].mean(0) for c in classes])
        nc = np.sqrt((cents * cents).sum(1, keepdims=True))
        cents = cents / np.where(nc > 1e-12, nc, 1.0)
        cen_hit += int((classes[(X[te] @ cents.T).argmax(1)] == y[te]).sum())
        total += int(te.sum())
    sil = _silhouette(X, y, np, rng)
    if total == 0:
        return None
    return {"knn": knn_hit / total, "centroid": cen_hit / total, "silhouette": sil,
            "n": int(total), "groups": int(len(ug))}


def _silhouette(X, y, np, rng, max_rows: int = 3000):
    """Mean silhouette in cosine distance over at most `max_rows` rows."""
    if len(X) > max_rows:
        sel = rng.choice(len(X), max_rows, replace=False)
        X, y = X[sel], y[sel]
    D = np.clip(1.0 - X @ X.T, 0.0, 2.0)
    classes = np.unique(y)
    if len(classes) < 2:
        return float("nan")
    own = np.zeros(len(X))
    other = np.full(len(X), np.inf)
    for c in classes:
        m = y == c
        cnt = m.sum()
        mean_to = D[:, m].sum(1) / np.maximum(cnt - m.astype(int), 1)
        own = np.where(m, mean_to, own)
        other = np.where(~m, np.minimum(other, D[:, m].mean(1)), other)
    s = (other - own) / np.maximum(np.maximum(own, other), 1e-12)
    _u, inv, cnt = np.unique(y, return_inverse=True, return_counts=True)
    s[cnt[inv] == 1] = 0.0                 # a singleton class: silhouette 0
    return float(s.mean())


__all__ = ["SOURCES", "BLOCKS", "NORMS", "DISTANCES", "SIMILARITIES", "REDUCES",
           "EmbeddingRef", "EmbeddingOption", "Embedding", "EmbeddingCache",
           "legacy_layer", "to_legacy_layer", "resolve_layer", "layer_name", "layers_of",
           "depth_of", "family_of", "families", "row_names", "family_label", "short_label",
           "net_options",
           "options", "option_for", "resolve", "signature_of", "distance", "similarity",
           "to_unit", "pair_block", "top_k", "project_2d", "separability"]
