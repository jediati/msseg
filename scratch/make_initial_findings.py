from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "scratch"
RUNS = SCRATCH / "runs"
PLOTS = SCRATCH / "plots"
REPORT = SCRATCH / "initial_findings.md"

DATASETS = [
    {
        "name": "labels.csv",
        "stem": "labels",
        "csv": Path(r"C:\Users\jediati\Desktop\JEDIATI\data\spears\labels.csv"),
    },
    {
        "name": "labels_2.csv",
        "stem": "labels_2",
        "csv": Path(r"C:\Users\jediati\Desktop\JEDIATI\data\spears\labels_2.csv"),
    },
]


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summary_table() -> str:
    rows = [
        "| Dataset | Area | Model | Balanced acc | ROC AUC | FG precision | FG recall | Errors |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for ds in DATASETS:
        for area_label in ["without_area", "with_area"]:
            run = load_json(RUNS / f"{ds['stem']}_{area_label}.json")
            for model in run["summary"]:
                errors = int(model["fp"]) + int(model["fn"])
                rows.append(
                    "| "
                    + " | ".join(
                        [
                            ds["name"],
                            "yes" if area_label == "with_area" else "no",
                            str(model["model"]),
                            pct(float(model["balanced_accuracy_mean"])),
                            pct(float(model["roc_auc_mean"])),
                            pct(float(model["fg_precision_mean"])),
                            pct(float(model["fg_recall_mean"])),
                            str(errors),
                        ]
                    )
                    + " |"
                )
    return "\n".join(rows)


def input_table() -> str:
    rows = [
        "| Input | Rows | Columns | Unlabeled | BG | FG | Features without area | Features with area | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for ds in DATASETS:
        df = pd.read_csv(ds["csv"])
        without_area = load_json(RUNS / f"{ds['stem']}_without_area.json")
        with_area = load_json(RUNS / f"{ds['stem']}_with_area.json")
        counts = df["class"].value_counts().to_dict()
        notes = "class 0 excluded"
        if "predicted" in df.columns:
            notes += "; predicted ignored"
        rows.append(
            "| "
            + " | ".join(
                [
                    ds["name"],
                    str(len(df)),
                    str(len(df.columns)),
                    str(int(counts.get(0, 0))),
                    str(int(counts.get(1, 0))),
                    str(int(counts.get(2, 0))),
                    str(without_area["feature_count"]),
                    str(with_area["feature_count"]),
                    notes,
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def hard_set_stats(ds: dict, disagreement_quantile: float = 0.90) -> tuple[pd.DataFrame, pd.DataFrame]:
    hard = pd.read_csv(RUNS / f"{ds['stem']}_with_area_hard_examples.csv")
    df = pd.read_csv(ds["csv"])
    labeled = df[df["class"].isin([1, 2])].copy()
    threshold = float(hard["model_disagreement"].quantile(disagreement_quantile))
    hard = hard[hard["model_disagreement"] >= threshold].copy()
    merged = hard.merge(
        labeled.reset_index(names="source_row"),
        on=["source_row", "slice", "region_id", "class"],
        how="left",
        suffixes=("", "_input"),
    )

    def stats(frame: pd.DataFrame, label: str) -> dict[str, str]:
        return {
            "set": label,
            "n": str(len(frame)),
            "bg": str(int((frame["class"] == 1).sum())),
            "fg": str(int((frame["class"] == 2).sum())),
            "median_area": f"{frame['area'].median():.0f}",
            "median_mean_base": f"{frame['mean_base'].median():.3f}",
            "median_ext_base": f"{frame['ext_base'].median():.3f}",
            "median_min_base": f"{frame['min_base'].median():.3f}",
            "median_max_base": f"{frame['max_base'].median():.3f}",
        }

    all_stats = stats(labeled, "all labeled")
    hard_stats = stats(merged, "top 10% disagreement")
    return merged, pd.DataFrame([all_stats, hard_stats])


def hard_stats_table(stats_by_dataset: dict[str, pd.DataFrame]) -> str:
    rows = [
        "| Dataset | Set | n | BG | FG | Median area | Median mean_base | Median ext_base | Median min_base | Median max_base |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, frame in stats_by_dataset.items():
        for _, row in frame.iterrows():
            rows.append(
                "| "
                + " | ".join(
                    [
                        name,
                        row["set"],
                        row["n"],
                        row["bg"],
                        row["fg"],
                        row["median_area"],
                        row["median_mean_base"],
                        row["median_ext_base"],
                        row["median_min_base"],
                        row["median_max_base"],
                    ]
                )
                + " |"
            )
    return "\n".join(rows)


def class_specific_hard_stats(ds: dict, hard: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(ds["csv"])
    labeled = df[df["class"].isin([1, 2])].copy()
    out = []
    for cls, class_name in [(1, "bg"), (2, "fg")]:
        for label, frame in [
            ("all labeled", labeled[labeled["class"] == cls]),
            ("top 10% disagreement", hard[hard["class"] == cls]),
        ]:
            out.append(
                {
                    "class": class_name,
                    "set": label,
                    "n": str(len(frame)),
                    "median_area": f"{frame['area'].median():.0f}",
                    "median_mean_base": f"{frame['mean_base'].median():.3f}",
                    "median_ext_base": f"{frame['ext_base'].median():.3f}",
                    "median_min_base": f"{frame['min_base'].median():.3f}",
                    "median_max_base": f"{frame['max_base'].median():.3f}",
                }
            )
    return pd.DataFrame(out)


def class_hard_stats_table(stats_by_dataset: dict[str, pd.DataFrame]) -> str:
    rows = [
        "| Dataset | Class | Set | n | Median area | Median mean_base | Median ext_base | Median min_base | Median max_base |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, frame in stats_by_dataset.items():
        for _, row in frame.iterrows():
            rows.append(
                "| "
                + " | ".join(
                    [
                        name,
                        row["class"],
                        row["set"],
                        row["n"],
                        row["median_area"],
                        row["median_mean_base"],
                        row["median_ext_base"],
                        row["median_min_base"],
                        row["median_max_base"],
                    ]
                )
                + " |"
            )
    return "\n".join(rows)


def make_plot(ds: dict, hard: pd.DataFrame, y_col: str) -> Path:
    df = pd.read_csv(ds["csv"])
    labeled = df[df["class"].isin([1, 2])].copy()
    hard_keys = set(zip(hard["source_row"], hard["region_id"]))

    fig, ax = plt.subplots(figsize=(6, 6))
    for cls, label, color in [(1, "bg", "#4C78A8"), (2, "fg", "#F58518")]:
        subset = labeled[labeled["class"] == cls]
        ax.scatter(
            np.log10(subset["area"].clip(lower=1)),
            subset[y_col],
            s=18,
            alpha=0.35,
            c=color,
            label=label,
            linewidths=0,
        )
    highlighted = labeled[
        [key in hard_keys for key in zip(labeled.index, labeled["region_id"])]
    ]
    ax.scatter(
        np.log10(highlighted["area"].clip(lower=1)),
        highlighted[y_col],
        s=58,
        facecolors="none",
        edgecolors="#D62728",
        linewidths=1.4,
        label="top 10% disagreement",
    )
    ax.set_box_aspect(1)
    ax.set_xlabel("log10(area)")
    ax.set_ylabel(y_col)
    ax.set_title(f"{ds['name']}: {y_col} vs area")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.2)
    out = PLOTS / f"{ds['stem']}_{y_col}_vs_area_disagreement.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def performance_notes() -> str:
    notes = []
    for ds in DATASETS:
        without_area = pd.DataFrame(load_json(RUNS / f"{ds['stem']}_without_area.json")["summary"])
        with_area = pd.DataFrame(load_json(RUNS / f"{ds['stem']}_with_area.json")["summary"])
        merged = with_area[["model", "balanced_accuracy_mean"]].merge(
            without_area[["model", "balanced_accuracy_mean"]],
            on="model",
            suffixes=("_with_area", "_without_area"),
        )
        merged["delta"] = merged["balanced_accuracy_mean_with_area"] - merged["balanced_accuracy_mean_without_area"]
        best = merged.iloc[merged["delta"].abs().argmax()]
        notes.append(
            f"- `{ds['name']}`: largest area effect was `{best['model']}` "
            f"({100 * best['delta']:+.1f} balanced-accuracy points)."
        )
    return "\n".join(notes)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    stats_by_dataset = {}
    class_stats_by_dataset = {}
    plot_paths = []
    hard_counts = []
    for ds in DATASETS:
        hard, stats = hard_set_stats(ds)
        stats_by_dataset[ds["name"]] = stats
        class_stats_by_dataset[ds["name"]] = class_specific_hard_stats(ds, hard)
        hard_counts.append(f"- `{ds['name']}`: top-disagreement threshold = {hard['model_disagreement'].min():.3f}; n = {len(hard)}.")
        plot_paths.extend([make_plot(ds, hard, "mean_base"), make_plot(ds, hard, "ext_base")])

    report = f"""# Initial Findings: Spears Label Classifiers

## Inputs

The supervised classification target is class `1` vs class `2`; class `0` is treated as unlabeled and excluded from training/evaluation. `slice`, `region_id`, bbox/position columns, and `predicted` when present are metadata, not model dimensions.

{input_table()}

## Model Performance With and Without Area

All results are 5-fold stratified cross-validation. Neural models report out-of-fold probabilities from PyTorch models with weighted cross entropy and internal validation early stopping. Random forest uses balanced class weights.

{summary_table()}

### Area Effect

{performance_notes()}

Random forest feature importance assigns very little direct weight to `area`: `0.16%` for `labels.csv` and `0.22%` for `labels_2.csv` in the with-area runs. Area still changed some neural outcomes, so it may be acting as a weak regularizing/context dimension rather than a primary separator.

## Hard-To-Explain Set

The hard-to-explain set is defined as the top 10% of labeled regions by cross-model `prob_fg` disagreement in the with-area run.

{chr(10).join(hard_counts)}

{hard_stats_table(stats_by_dataset)}

Class-specific view:

{class_hard_stats_table(class_stats_by_dataset)}

Interpretation: the disagreement set is not explained cleanly by `area` alone. Median areas are close to the full labeled populations, and the random forest gives area almost no importance. The stronger pattern is in `base_` intensity. In `labels.csv`, hard bg rows are brighter than typical bg and hard fg rows are less bright than typical fg, so both move toward the class boundary in base-intensity space. In `labels_2.csv`, hard fg rows are strongly bg-like by `mean_base`/`ext_base`, while hard bg rows are even darker than the full bg set, suggesting extreme-background or boundary cases. That points to ambiguous/transitional material or label-boundary issues more than to a missing deep-CNN architecture.

## Four Square Plots

Each plot is square. Filled points are all labeled rows; red outlines mark the top 10% disagreement rows.

![labels.csv mean_base vs area](plots/labels_mean_base_vs_area_disagreement.png)

![labels.csv ext_base vs area](plots/labels_ext_base_vs_area_disagreement.png)

![labels_2.csv mean_base vs area](plots/labels_2_mean_base_vs_area_disagreement.png)

![labels_2.csv ext_base vs area](plots/labels_2_ext_base_vs_area_disagreement.png)

## Files

- Run outputs: `scratch/runs/`
- Probability CSVs: `scratch/runs/*_predictions.csv`
- Hard-example CSVs: `scratch/runs/*_hard_examples.csv`
- Plot images: `scratch/plots/`
"""
    REPORT.write_text(report, encoding="utf-8")
    print(f"wrote {REPORT}")
    for path in plot_paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
