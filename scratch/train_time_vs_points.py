from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from compare_spears_labels_models import (
    DEFAULT_CSV,
    IGNORED_COLUMNS,
    make_torch_model,
    set_seeds,
)


def subsample_indices(y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    if fraction >= 1.0:
        return np.arange(len(y))
    idx, _ = train_test_split(
        np.arange(len(y)),
        train_size=fraction,
        stratify=y,
        random_state=seed,
    )
    return np.sort(idx)


def train_rf(x_train: np.ndarray, y_train: np.ndarray, trees: int, seed: int):
    scaler = StandardScaler().fit(x_train)
    model = RandomForestClassifier(
        n_estimators=trees,
        random_state=seed,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=-1,
    )
    t0 = time.perf_counter()
    model.fit(scaler.transform(x_train), y_train)
    train_sec = time.perf_counter() - t0
    return model, scaler, train_sec


def predict_rf(model, scaler, x: np.ndarray) -> tuple[np.ndarray, float]:
    xs = scaler.transform(x)
    t0 = time.perf_counter()
    probs = model.predict_proba(xs)[:, 1]
    classify_sec = time.perf_counter() - t0
    return probs.astype(np.float64), classify_sec


def train_torch(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    max_epochs: int,
    patience: int,
):
    set_seeds(seed)
    if len(np.unique(y_train)) < 2:
        raise ValueError("Need both classes in the training subsample.")
    val_size = 0.2 if len(y_train) >= 20 else max(1, len(y_train) // 5) / len(y_train)
    train_idx, val_idx = train_test_split(
        np.arange(len(y_train)),
        test_size=val_size,
        stratify=y_train,
        random_state=seed,
    )
    scaler = StandardScaler().fit(x_train[train_idx])
    x_scaled = scaler.transform(x_train).astype(np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_scaled[train_idx]), torch.from_numpy(y_train[train_idx])),
        batch_size=min(32, max(1, len(train_idx))),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = make_torch_model(model_name, x_train.shape[1])
    counts = np.bincount(y_train[train_idx], minlength=2).astype(np.float32)
    weights = counts.sum() / (2 * np.maximum(counts, 1))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    x_val = torch.from_numpy(x_scaled[val_idx])
    y_val = y_train[val_idx]
    best_state = None
    best_val = -1.0
    stale = 0
    best_epoch = 0

    t0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_probs = torch.softmax(model(x_val), dim=1)[:, 1].numpy()
        val_score = balanced_accuracy_score(y_val, (val_probs >= 0.5).astype(np.int64))
        if val_score > best_val + 1e-4:
            best_val = float(val_score)
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    train_sec = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler, train_sec, best_epoch


def predict_torch(model: nn.Module, scaler: StandardScaler, x: np.ndarray) -> tuple[np.ndarray, float]:
    xs = torch.from_numpy(scaler.transform(x).astype(np.float32))
    model.eval()
    # Warmup so first CUDA/CPU graph cost is not counted if present.
    with torch.no_grad():
        _ = model(xs[: min(8, len(xs))])
    t0 = time.perf_counter()
    with torch.no_grad():
        probs = torch.softmax(model(xs), dim=1)[:, 1].numpy()
    classify_sec = time.perf_counter() - t0
    return probs.astype(np.float64), classify_sec


def score(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    pred = (probs >= 0.5).astype(np.int64)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) == 2 else float("nan"),
    }


def run_one_fraction(
    *,
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    trees: int,
    epochs: int,
    patience: int,
    classify_repeats: int,
) -> dict[str, float | int | str]:
    if model_name == "random_forest":
        model, scaler, train_sec = train_rf(x_train, y_train, trees, seed)
        classify_secs = []
        probs = None
        for r in range(classify_repeats):
            probs, csec = predict_rf(model, scaler, x_test)
            classify_secs.append(csec)
        assert probs is not None
        metrics = score(y_test, probs)
        return {
            "model": model_name,
            "train_sec": float(train_sec),
            "classify_sec_total": float(np.mean(classify_secs)),
            "classify_sec_per_point": float(np.mean(classify_secs) / max(len(x_test), 1)),
            "best_epoch": -1,
            **metrics,
        }

    model, scaler, train_sec, best_epoch = train_torch(
        model_name, x_train, y_train, seed, epochs, patience
    )
    classify_secs = []
    probs = None
    for _ in range(classify_repeats):
        probs, csec = predict_torch(model, scaler, x_test)
        classify_secs.append(csec)
    assert probs is not None
    metrics = score(y_test, probs)
    return {
        "model": model_name,
        "train_sec": float(train_sec),
        "classify_sec_total": float(np.mean(classify_secs)),
        "classify_sec_per_point": float(np.mean(classify_secs) / max(len(x_test), 1)),
        "best_epoch": int(best_epoch),
        **metrics,
    }


def plot_curves(frame: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6.5))
    colors = {
        "random_forest": "#4C78A8",
        "dense_fc": "#F58518",
        "cnn2": "#54A24B",
        "cnn3": "#E45756",
    }
    markers = {
        "random_forest": "o",
        "dense_fc": "s",
        "cnn2": "^",
        "cnn3": "D",
    }
    for model, group in frame.groupby("model"):
        g = group.sort_values("n_train")
        ax.plot(
            g["n_train"],
            g["train_sec"],
            color=colors.get(model, "gray"),
            marker=markers.get(model, "o"),
            linewidth=2,
            markersize=7,
            label=model,
        )
        for _, row in g.iterrows():
            label = f"bal {100 * row['balanced_accuracy']:.0f}%\nroc {100 * row['roc_auc']:.0f}%"
            ax.annotate(
                label,
                (row["n_train"], row["train_sec"]),
                textcoords="offset points",
                xytext=(8, 6),
                fontsize=8,
                color=colors.get(model, "black"),
            )
    ax.set_xlabel("Labeled training points")
    ax.set_ylabel("Train time (sec)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-time vs labeled-point-count curves.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-json", type=Path, default=Path("scratch/train_time_vs_points.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("scratch/train_time_vs_points.csv"))
    parser.add_argument("--out-plot", type=Path, default=Path("scratch/plots/train_time_vs_points.png"))
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--repeats", type=int, default=3, help="Random subsample repeats per fraction.")
    parser.add_argument("--classify-repeats", type=int, default=5)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-area", action="store_true")
    args = parser.parse_args()

    set_seeds(args.seed)
    df = pd.read_csv(args.csv)
    features = [c for c in df.columns if c not in set(IGNORED_COLUMNS)]
    if args.exclude_area:
        features = [c for c in features if c != "area"]
    labeled = df[df["class"].isin([1, 2])].copy()
    x_all = labeled[features].to_numpy(dtype=np.float32)
    y_all = (labeled["class"].to_numpy() == 2).astype(np.int64)

    # Fixed held-out test set so quality labels are comparable across train sizes.
    train_pool_idx, test_idx = train_test_split(
        np.arange(len(y_all)),
        test_size=args.test_size,
        stratify=y_all,
        random_state=args.seed,
    )
    x_test = x_all[test_idx]
    y_test = y_all[test_idx]
    x_pool = x_all[train_pool_idx]
    y_pool = y_all[train_pool_idx]

    models = ["random_forest", "dense_fc", "cnn2", "cnn3"]
    rows: list[dict[str, float | int | str]] = []

    for fraction in args.fractions:
        for repeat in range(args.repeats):
            seed = args.seed + 1000 * int(100 * fraction) + repeat
            sub_idx = subsample_indices(y_pool, fraction, seed)
            x_train = x_pool[sub_idx]
            y_train = y_pool[sub_idx]
            print(
                f"fraction={fraction:.2f} repeat={repeat+1}/{args.repeats} "
                f"n_train={len(y_train)} n_test={len(y_test)}",
                flush=True,
            )
            for model_name in models:
                result = run_one_fraction(
                    model_name=model_name,
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    seed=seed,
                    trees=args.trees,
                    epochs=args.epochs,
                    patience=args.patience,
                    classify_repeats=args.classify_repeats,
                )
                result.update(
                    {
                        "fraction": float(fraction),
                        "repeat": int(repeat),
                        "n_train": int(len(y_train)),
                        "n_test": int(len(y_test)),
                        "n_pool": int(len(y_pool)),
                        "n_labeled_total": int(len(y_all)),
                    }
                )
                rows.append(result)
                print(
                    f"  {model_name}: train={result['train_sec']:.3f}s "
                    f"classify={1e6 * float(result['classify_sec_per_point']):.1f} us/pt "
                    f"bal={100 * float(result['balanced_accuracy']):.1f}% "
                    f"roc={100 * float(result['roc_auc']):.1f}%",
                    flush=True,
                )

    frame = pd.DataFrame(rows)
    # Mean across random subsample repeats for the curve.
    curve = (
        frame.groupby(["model", "fraction", "n_train"], as_index=False)
        .agg(
            train_sec=("train_sec", "mean"),
            train_sec_std=("train_sec", "std"),
            classify_sec_total=("classify_sec_total", "mean"),
            classify_sec_per_point=("classify_sec_per_point", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            roc_auc=("roc_auc", "mean"),
        )
        .sort_values(["model", "n_train"])
    )
    classify_summary = (
        frame.groupby("model", as_index=False)
        .agg(
            classify_sec_total_mean=("classify_sec_total", "mean"),
            classify_sec_per_point_mean=("classify_sec_per_point", "mean"),
            classify_sec_per_point_std=("classify_sec_per_point", "std"),
        )
        .sort_values("classify_sec_per_point_mean")
    )

    out = {
        "csv_path": str(args.csv),
        "feature_count": len(features),
        "area_included": not args.exclude_area,
        "fractions": args.fractions,
        "repeats": args.repeats,
        "test_size": args.test_size,
        "n_labeled_total": int(len(y_all)),
        "n_pool": int(len(y_pool)),
        "n_test": int(len(y_test)),
        "method": (
            "Fixed stratified held-out test set; for each fraction, randomly subsample "
            "the train pool (stratified) and retrain each model. Train time is wall-clock "
            "fit/early-stopped training. Classify time averages repeated full-test-set "
            "forward passes after a warmup."
        ),
        "raw": rows,
        "curve": curve.to_dict(orient="records"),
        "classify_summary": classify_summary.to_dict(orient="records"),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_plot.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    curve.to_csv(args.out_csv, index=False)
    classify_summary.to_csv(args.out_csv.with_name(args.out_csv.stem + "_classify.csv"), index=False)
    plot_curves(
        curve,
        args.out_plot,
        title=f"Train time vs labeled points ({args.csv.name})",
    )

    print("\ncurve:")
    print(curve.to_string(index=False))
    print("\nclassify summary:")
    print(classify_summary.to_string(index=False))
    print(f"\nwrote {args.out_json}")
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_plot}")


if __name__ == "__main__":
    main()
