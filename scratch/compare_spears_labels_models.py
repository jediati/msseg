from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_CSV = Path(r"C:\Users\jediati\Desktop\JEDIATI\data\spears\labels.csv")
IGNORED_COLUMNS = [
    "slice",
    "region_id",
    "class",
    "predicted",
    "bbox_w",
    "bbox_h",
    "min_x",
    "max_x",
    "min_y",
    "max_y",
]


class DenseNet(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvNet1D(nn.Module):
    def __init__(self, n_features: int, n_conv_layers: int) -> None:
        super().__init__()
        channels = [1, 16, 32, 64][: n_conv_layers + 1]
        blocks: list[nn.Module] = []
        for i in range(n_conv_layers):
            blocks.extend(
                [
                    nn.Conv1d(channels[i], channels[i + 1], kernel_size=3, padding=1),
                    nn.BatchNorm1d(channels[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(0.10),
                ]
            )
        self.conv = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The CSV is tabular, so the CNN treats the ordered feature vector as 1D.
        return self.head(self.conv(x.unsqueeze(1)))


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def metrics_from_probs(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float | int]:
    pred = (probs >= 0.5).astype(np.int64)
    precision, recall, f1_each, _ = precision_recall_fscore_support(
        y_true, pred, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "bg_precision": float(precision[0]),
        "bg_recall": float(recall[0]),
        "bg_f1": float(f1_each[0]),
        "fg_precision": float(precision[1]),
        "fg_recall": float(recall[1]),
        "fg_f1": float(f1_each[1]),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def make_torch_model(model_name: str, n_features: int) -> nn.Module:
    if model_name == "dense_fc":
        return DenseNet(n_features)
    if model_name == "cnn2":
        return ConvNet1D(n_features, 2)
    if model_name == "cnn3":
        return ConvNet1D(n_features, 3)
    raise ValueError(f"Unknown torch model: {model_name}")


def train_torch_model(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    max_epochs: int,
    patience: int,
) -> tuple[dict[str, float | int], np.ndarray]:
    set_seeds(seed)
    train_idx, val_idx = train_test_split(
        np.arange(len(y_train)),
        test_size=0.2,
        stratify=y_train,
        random_state=seed,
    )
    scaler = StandardScaler().fit(x_train[train_idx])
    x_train_scaled = scaler.transform(x_train).astype(np.float32)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)

    train_data = TensorDataset(
        torch.from_numpy(x_train_scaled[train_idx]),
        torch.from_numpy(y_train[train_idx]),
    )
    train_loader = DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    model = make_torch_model(model_name, x_train.shape[1])
    class_counts = np.bincount(y_train[train_idx], minlength=2).astype(np.float32)
    class_weights = class_counts.sum() / (2 * np.maximum(class_counts, 1))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    x_val = torch.from_numpy(x_train_scaled[val_idx])
    y_val = y_train[val_idx]
    best_state = None
    best_epoch = 0
    best_val = -1.0
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_probs = torch.softmax(model(x_val), dim=1)[:, 1].numpy()
        val_score = balanced_accuracy_score(y_val, (val_probs >= 0.5).astype(np.int64))
        if val_score > best_val + 1e-4:
            best_val = float(val_score)
            best_epoch = epoch
            stale_epochs = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(x_test_scaled)), dim=1)[:, 1].numpy()
    out = metrics_from_probs(y_test, probs)
    out["best_epoch"] = int(best_epoch)
    out["val_balanced_accuracy"] = float(best_val)
    return out, probs


def feature_group(feature: str) -> str:
    if feature.startswith("ext_"):
        return "extremum"
    if "blur" in feature:
        return "blur"
    if "edges" in feature:
        return "edges"
    if "laplacian" in feature:
        return "laplacian"
    if feature.endswith("_base") or feature in {
        "mean_base",
        "min_base",
        "max_base",
        "std_base",
        "relevance_base",
    }:
        return "base"
    return "other"


def append_predictions(
    rows: list[dict[str, float | int | str]],
    labeled: pd.DataFrame,
    test_idx: np.ndarray,
    y_test: np.ndarray,
    model: str,
    fold: int,
    probs: np.ndarray,
) -> None:
    pred_fg = probs >= 0.5
    for local_i, true_is_fg, prob_fg, predicted_is_fg in zip(test_idx, y_test, probs, pred_fg):
        source = labeled.iloc[int(local_i)]
        predicted_class = 2 if bool(predicted_is_fg) else 1
        true_class = int(source["class"])
        rows.append(
            {
                "source_row": int(source.name),
                "slice": str(source["slice"]) if "slice" in labeled.columns else "",
                "region_id": int(source["region_id"]) if "region_id" in labeled.columns else int(local_i),
                "class": true_class,
                "model": model,
                "fold": int(fold),
                "prob_fg": float(prob_fg),
                "prob_bg": float(1.0 - prob_fg),
                "predicted_class": predicted_class,
                "confidence": float(max(prob_fg, 1.0 - prob_fg)),
                "margin_from_0_5": float(abs(prob_fg - 0.5)),
                "correct": int(predicted_class == true_class),
            }
        )


def summarize_folds(folds: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    frame = pd.DataFrame(folds)
    summary = []
    for model, group in frame.groupby("model"):
        row: dict[str, float | int | str] = {"model": model}
        for metric in [
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "roc_auc",
            "bg_recall",
            "fg_recall",
            "fg_precision",
            "fg_f1",
        ]:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        for name in ["tn", "fp", "fn", "tp"]:
            row[name] = int(group[name].sum())
        summary.append(row)
    return sorted(summary, key=lambda r: float(r["balanced_accuracy_mean"]), reverse=True)


def make_hard_examples(predictions: pd.DataFrame) -> pd.DataFrame:
    wide = predictions.pivot_table(
        index=["source_row", "slice", "region_id", "class"],
        columns="model",
        values="prob_fg",
    ).reset_index()
    wide.columns.name = None
    stats = (
        predictions.groupby(["source_row", "slice", "region_id", "class"])
        .agg(
            mean_confidence=("confidence", "mean"),
            min_confidence=("confidence", "min"),
            model_disagreement=("prob_fg", lambda s: float(s.max() - s.min())),
            n_correct=("correct", "sum"),
            n_models=("correct", "count"),
        )
        .reset_index()
    )
    out = wide.merge(stats, on=["source_row", "slice", "region_id", "class"])
    out["hardness_rank_key"] = (
        (out["n_models"] - out["n_correct"]) * 10
        + (1 - out["min_confidence"])
        + out["model_disagreement"]
    )
    return out.sort_values(["hardness_rank_key", "model_disagreement"], ascending=[False, False])


def run(args: argparse.Namespace) -> dict[str, object]:
    set_seeds(args.seed)
    df = pd.read_csv(args.csv)
    feature_columns = [c for c in df.columns if c not in set(IGNORED_COLUMNS)]
    if args.exclude_area:
        feature_columns = [c for c in feature_columns if c != "area"]
    labeled = df[df["class"].isin([1, 2])].copy()
    x = labeled[feature_columns].to_numpy(dtype=np.float32)
    y = (labeled["class"].to_numpy() == 2).astype(np.int64)

    folds: list[dict[str, float | int | str]] = []
    oof_predictions: list[dict[str, float | int | str]] = []
    rf_importances = []
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    start = time.time()

    for fold, (train_idx, test_idx) in enumerate(skf.split(x, y), start=1):
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        scaler = StandardScaler().fit(x_train)
        rf = RandomForestClassifier(
            n_estimators=args.trees,
            random_state=args.seed + fold,
            class_weight="balanced",
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=-1,
        )
        rf.fit(scaler.transform(x_train), y_train)
        rf_probs = rf.predict_proba(scaler.transform(x_test))[:, 1]
        rf_result = metrics_from_probs(y_test, rf_probs)
        rf_result.update({"model": "random_forest", "fold": fold})
        folds.append(rf_result)
        rf_importances.append(rf.feature_importances_)
        append_predictions(oof_predictions, labeled, test_idx, y_test, "random_forest", fold, rf_probs)

        for model_name in ["dense_fc", "cnn2", "cnn3"]:
            result, probs = train_torch_model(
                model_name,
                x_train,
                y_train,
                x_test,
                y_test,
                args.seed + 100 * fold + len(folds),
                args.epochs,
                args.patience,
            )
            result.update({"model": model_name, "fold": fold})
            folds.append(result)
            append_predictions(oof_predictions, labeled, test_idx, y_test, model_name, fold, probs)

        print(f"finished fold {fold}/{args.folds}", flush=True)

    importances = np.vstack(rf_importances)
    mean_importance = importances.mean(axis=0)
    std_importance = importances.std(axis=0, ddof=1) if len(importances) > 1 else np.zeros_like(mean_importance)
    top_features = sorted(
        [
            {"feature": feature, "importance": float(mean), "std": float(std)}
            for feature, mean, std in zip(feature_columns, mean_importance, std_importance)
        ],
        key=lambda row: row["importance"],
        reverse=True,
    )[: args.top_features]

    group_importance: dict[str, float] = {}
    for feature, importance in zip(feature_columns, mean_importance):
        group = feature_group(feature)
        group_importance[group] = group_importance.get(group, 0.0) + float(importance)
    feature_groups = sorted(
        [{"group": group, "importance": importance} for group, importance in group_importance.items()],
        key=lambda row: row["importance"],
        reverse=True,
    )

    return {
        "csv_path": str(args.csv),
        "shape": list(df.shape),
        "ignored_columns": IGNORED_COLUMNS,
        "area_included": not args.exclude_area,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "class_counts_all": {str(k): int(v) for k, v in df["class"].value_counts().sort_index().items()},
        "class_counts_supervised": {
            "bg_1": int((labeled["class"] == 1).sum()),
            "fg_2": int((labeled["class"] == 2).sum()),
        },
        "method": (
            f"{args.folds}-fold stratified CV; class 0 excluded as unlabeled; "
            f"area {'included' if not args.exclude_area else 'excluded'}; "
            "features standardized per fold for all models; RF uses balanced class weights; "
            "PyTorch neural nets use AdamW, weighted cross entropy, and internal validation early stopping."
        ),
        "folds": folds,
        "summary": summarize_folds(folds),
        "top_features": top_features,
        "feature_groups": feature_groups,
        "oof_predictions": oof_predictions,
        "elapsed_sec": time.time() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare classifiers for spears labels.csv.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=Path("scratch/spears_model_comparison.json"))
    parser.add_argument("--summary-csv", type=Path, default=Path("scratch/spears_model_summary.csv"))
    parser.add_argument("--predictions-csv", type=Path, default=Path("scratch/spears_model_predictions.csv"))
    parser.add_argument("--hard-examples-csv", type=Path, default=Path("scratch/spears_model_hard_examples.csv"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--top-features", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-area", action="store_true")
    args = parser.parse_args()

    result = run(args)
    oof_predictions = result.pop("oof_predictions")
    predictions = pd.DataFrame(oof_predictions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    args.hard_examples_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(result["summary"]).to_csv(args.summary_csv, index=False)
    predictions.to_csv(args.predictions_csv, index=False)
    make_hard_examples(predictions).to_csv(args.hard_examples_csv, index=False)

    print(json.dumps(result["summary"], indent=2))
    print(f"\nwrote {args.out}")
    print(f"wrote {args.summary_csv}")
    print(f"wrote {args.predictions_csv}")
    print(f"wrote {args.hard_examples_csv}")


if __name__ == "__main__":
    main()
