"""The ``train`` sub-command, shared by every harvest driver.

    <driver> train HARVEST_DIR --out encoder.msenc [--dim 8] [--arch mlp|pca] ...

``add_train_arguments`` fills an argparse (sub)parser; ``run_train`` runs it
and writes the bundle. A driver (``mspath-embed``) owns the top-level parser
and its own ``harvest`` sub-command.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable, Optional

from .shards import HarvestReader
from .train import ARCHS, DEFAULT_LOG_COLUMNS, TrainSettings, train


def _ints(text: str):
    return tuple(int(v) for v in str(text).replace(";", ",").split(",") if v.strip())


def _strs(text: str):
    return tuple(v.strip() for v in str(text).split(",") if v.strip())


def add_train_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("harvest", help="a harvest directory (harvest.json + shards/)")
    p.add_argument("--out", required=True, help="the encoder bundle to write (.msenc)")
    p.add_argument("--dim", type=int, default=8, help="latent width (default 8)")
    p.add_argument("--arch", choices=ARCHS, default="mlp",
                   help="mlp (the walk objective, needs torch) or pca (trial zero, numpy)")
    p.add_argument("--hidden", type=_ints, default=(64, 32), help="hidden widths, e.g. 64,32")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--walk", type=int, default=2, help="max random-walk hops for a positive")
    p.add_argument("--temp", type=float, default=0.1, help="InfoNCE temperature")
    p.add_argument("--rec", type=float, default=0.1, help="reconstruction weight")
    p.add_argument("--vc", type=float, default=0.05, help="variance + covariance weight")
    p.add_argument("--group-drop", type=float, default=0.15,
                   help="probability a column group is dropped from a view")
    p.add_argument("--corrupt", type=float, default=0.1,
                   help="probability a column is replaced from another row")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0")
    p.add_argument("--max-rows", type=int, default=None,
                   help="bound the rows loaded (a seeded subset of whole shards)")
    p.add_argument("--factors", type=str, default=None,
                   help="use only these persistence factors of the harvest, e.g. 1,2")
    p.add_argument("--log-columns", type=_strs, default=DEFAULT_LOG_COLUMNS,
                   help="columns to log1p before standardising (default area,bbox_w,bbox_h)")
    p.add_argument("--drop", type=_strs, default=(),
                   help="extra columns to leave out of the encoder's input")
    p.add_argument("--history", default=None,
                   help="also write the per-epoch history as JSON to this path")
    p.add_argument("--quiet", action="store_true")
    return p


def run_train(args: argparse.Namespace, log: Optional[Callable[[str], None]] = None) -> int:
    say = log or ((lambda _m: None) if args.quiet else (lambda m: print(m, file=sys.stderr)))
    reader = HarvestReader(args.harvest)
    factors = None
    if args.factors:
        factors = [float(v) for v in str(args.factors).split(",") if v.strip()]
    harvest = reader.load(max_rows=args.max_rows, seed=args.seed, factors=factors)
    say(f"harvest {args.harvest}: {len(harvest.shards)} shards, {harvest.n_rows} rows, "
        f"{harvest.n_arcs} arcs")
    settings = TrainSettings(dim=args.dim, arch=args.arch, hidden=tuple(args.hidden),
                             epochs=args.epochs, batch=args.batch, lr=args.lr,
                             weight_decay=args.weight_decay, walk=args.walk, temp=args.temp,
                             rec=args.rec, vc=args.vc, group_drop=args.group_drop,
                             corrupt=args.corrupt, seed=args.seed, device=args.device,
                             log_columns=tuple(args.log_columns), extra_drop=tuple(args.drop))
    bundle, history = train(harvest, settings, log=say)
    out = str(args.out)
    if not out.lower().endswith(".msenc"):
        out += ".msenc"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    bundle.save(out)
    if args.history:
        with open(args.history, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=1)
    say(f"wrote {out}: {bundle.describe()}; columns {bundle.column_names()[0]} .. "
        f"{bundle.column_names()[-1]}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="msseg-embed-train",
                                description="Train a task-free region encoder on a harvest.")
    add_train_arguments(p)
    return run_train(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
