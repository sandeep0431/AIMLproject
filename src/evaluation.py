from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
    matthews_corrcoef,
    cohen_kappa_score,
)
from sklearn.preprocessing import label_binarize

from src.data_utils import ORDERED_CLASSES
from src.ordinal_cost import align_probabilities


def evaluate_predictions(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_proba: Optional[np.ndarray] = None,
    estimator_classes: Optional[np.ndarray | List[str]] = None,
    model_name: str = "Model",
) -> Dict[str, Any]:
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    macro_prec = float(precision_score(y_true_arr, y_pred_arr, labels=ORDERED_CLASSES, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true_arr, y_pred_arr, labels=ORDERED_CLASSES, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, labels=ORDERED_CLASSES, average="macro", zero_division=0))

    weighted_prec = float(precision_score(y_true_arr, y_pred_arr, labels=ORDERED_CLASSES, average="weighted", zero_division=0))
    weighted_rec = float(recall_score(y_true_arr, y_pred_arr, labels=ORDERED_CLASSES, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, labels=ORDERED_CLASSES, average="weighted", zero_division=0))

    high_risk_label = "high risk"
    high_risk_prec = float(precision_score(y_true_arr, y_pred_arr, labels=[high_risk_label], average=None, zero_division=0)[0])
    high_risk_rec = float(recall_score(y_true_arr, y_pred_arr, labels=[high_risk_label], average=None, zero_division=0)[0])
    high_risk_f1 = float(f1_score(y_true_arr, y_pred_arr, labels=[high_risk_label], average=None, zero_division=0)[0])

    per_class = {}
    for cls in ORDERED_CLASSES:
        p = float(precision_score(y_true_arr, y_pred_arr, labels=[cls], average=None, zero_division=0)[0])
        r = float(recall_score(y_true_arr, y_pred_arr, labels=[cls], average=None, zero_division=0)[0])
        f = float(f1_score(y_true_arr, y_pred_arr, labels=[cls], average=None, zero_division=0)[0])
        s = int(np.sum(y_true_arr == cls))
        per_class[cls] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "support": s,
        }

    # agreement and correlation metrics for multi-class ordinal evaluation
    mcc = float(matthews_corrcoef(y_true_arr, y_pred_arr))
    kappa = float(cohen_kappa_score(y_true_arr, y_pred_arr))

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=ORDERED_CLASSES)

    clf_rep = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=ORDERED_CLASSES,
        target_names=ORDERED_CLASSES,
        digits=4,
        zero_division=0,
    )

    roc_auc_macro = None
    per_class_roc = {}
    aligned_proba = None
    if y_proba is not None and estimator_classes is not None:
        try:
            aligned_proba = align_probabilities(y_proba, estimator_classes, ORDERED_CLASSES)
            y_bin = label_binarize(y_true_arr, classes=ORDERED_CLASSES)
            # multiclass roc-auc evaluated as one-vs-rest macro average
            roc_auc_macro = float(roc_auc_score(y_bin, aligned_proba, multi_class="ovr", average="macro"))

            for idx, cls in enumerate(ORDERED_CLASSES):
                fpr, tpr, _ = roc_curve(y_bin[:, idx], aligned_proba[:, idx])
                per_class_roc[cls] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "auc": float(auc(fpr, tpr)),
                }
        except Exception as e:
            roc_auc_macro = None
            per_class_roc = {"error": str(e)}

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_prec, 4),
        "weighted_recall": round(weighted_rec, 4),
        "weighted_f1": round(weighted_f1, 4),
        "high_risk_precision": round(high_risk_prec, 4),
        "high_risk_recall": round(high_risk_rec, 4),
        "high_risk_f1": round(high_risk_f1, 4),
        "mcc": round(mcc, 4),
        "cohen_kappa": round(kappa, 4),
        "roc_auc_macro": round(roc_auc_macro, 4) if roc_auc_macro is not None else "N/A",
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_rep,
        "aligned_proba": aligned_proba,
        "per_class_roc": per_class_roc,
    }


def plot_confusion_matrix(cm_data: list | np.ndarray, model_slug: str, title: str, output_dir: Path) -> Path:
    cm = np.array(cm_data)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=ORDERED_CLASSES,
        yticklabels=ORDERED_CLASSES,
        cbar=True,
    )
    plt.title(f"Confusion Matrix: {title}", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("Predicted Class", fontsize=10)
    plt.ylabel("True Class", fontsize=10)
    plt.tight_layout()
    out_path = output_dir / f"confusion_matrix_{model_slug}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_roc_curves(per_class_roc: Dict[str, Any], model_slug: str, title: str, output_dir: Path) -> Optional[Path]:
    if not per_class_roc or "error" in per_class_roc:
        return None

    plt.figure(figsize=(6.5, 5))
    colors = {"low risk": "#2ecc71", "mid risk": "#f39c12", "high risk": "#e74c3c"}
    for cls in ORDERED_CLASSES:
        if cls in per_class_roc:
            fpr = np.array(per_class_roc[cls]["fpr"])
            tpr = np.array(per_class_roc[cls]["tpr"])
            roc_score = per_class_roc[cls]["auc"]
            plt.plot(fpr, tpr, label=f"{cls} (AUC = {roc_score:.3f})", color=colors.get(cls, "blue"), lw=2)

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate", fontsize=10)
    plt.ylabel("True Positive Rate", fontsize=10)
    plt.title(f"ROC Curves (OvR): {title}", fontsize=12, fontweight="bold", pad=10)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = output_dir / f"roc_curves_{model_slug}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def generate_comparison_charts(results: List[Dict[str, Any]], output_dir: Path):
    df_res = pd.DataFrame(results)

    plt.figure(figsize=(9, 4.8))
    bars = plt.barh(df_res["model_name"], df_res["accuracy"], color="#3498db", edgecolor="#2980b9")
    plt.xlim(0.5, 1.0)
    plt.xlabel("Test Accuracy", fontsize=11)
    plt.title("Model Comparison — Test Accuracy", fontsize=13, fontweight="bold", pad=12)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=9, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    y = np.arange(len(df_res))
    height = 0.35
    plt.barh(y - height / 2, df_res["macro_f1"], height, label="Macro-F1", color="#9b59b6", edgecolor="#8e44ad")
    plt.barh(y + height / 2, df_res["weighted_f1"], height, label="Weighted-F1", color="#1abc9c", edgecolor="#16a085")
    plt.yticks(y, df_res["model_name"])
    plt.xlim(0.5, 1.0)
    plt.xlabel("F1-Score", fontsize=11)
    plt.title("Model Comparison — Macro-F1 vs Weighted-F1", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "f1_comparison.png", dpi=300)
    plt.close()

    class_metric_rows = []
    for r in results:
        for cls in ORDERED_CLASSES:
            class_metric_rows.append({
                "Model": r["model_name"],
                "Class": cls,
                "Precision": r["per_class"][cls]["precision"],
                "Recall": r["per_class"][cls]["recall"],
                "F1": r["per_class"][cls]["f1"],
            })
    df_class = pd.DataFrame(class_metric_rows)
    df_class.to_csv(output_dir.parent / "per_class_metrics.csv", index=False)

    plt.figure(figsize=(12, 5.5))
    sns.barplot(data=df_class, x="Model", y="Recall", hue="Class", palette={"low risk": "#2ecc71", "mid risk": "#f39c12", "high risk": "#e74c3c"})
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0.4, 1.05)
    plt.title("Class-Specific Recall Comparison Across Models", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Recall (Sensitivity)", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "per_class_metrics.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6.5))
    metrics_to_plot = ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "high_risk_recall"]
    metric_labels = ["Accuracy", "Macro-F1", "Weighted-F1", "Balanced Acc", "High-Risk Recall"]
    
    x = np.arange(len(df_res))
    width = 0.15
    colors = ["#3498db", "#9b59b6", "#1abc9c", "#e67e22", "#e74c3c"]

    for i, (m_col, m_name, color) in enumerate(zip(metrics_to_plot, metric_labels, colors)):
        plt.bar(x + (i - 2) * width, df_res[m_col], width, label=m_name, color=color, edgecolor="black", alpha=0.9)

    plt.xticks(x, df_res["model_name"], rotation=25, ha="right", fontsize=9)
    plt.ylim(0.45, 1.02)
    plt.ylabel("Score", fontsize=11)
    plt.title("Maternal Health Risk — Comprehensive Evaluation Dashboard", fontsize=14, fontweight="bold", pad=14)
    plt.legend(loc="lower right", ncol=3)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_dashboard.png", dpi=300)
    plt.close()


def generate_ablation_chart(ablation_df: pd.DataFrame, output_dir: Path):
    plt.figure(figsize=(10, 5))
    y = np.arange(len(ablation_df))
    height = 0.35
    plt.barh(y - height / 2, ablation_df["macro_f1"], height, label="Macro-F1", color="#2980b9", edgecolor="#1c5980")
    plt.barh(y + height / 2, ablation_df["high_risk_recall"], height, label="High-Risk Recall", color="#e74c3c", edgecolor="#c0392b")
    plt.yticks(y, ablation_df["ablation_version"] + ": " + ablation_df["description"])
    plt.xlim(0.5, 1.02)
    plt.xlabel("Score", fontsize=11)
    plt.title("Controlled Ablation Progression (V0 - V7)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_comparison.png", dpi=300)
    plt.close()
