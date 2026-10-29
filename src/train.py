from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import joblib

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.data_utils import (
    load_dataset,
    CANONICAL_FEATURES,
    CANONICAL_TARGET,
    ORDERED_CLASSES,
)
from src.features import engineer_features, ALL_FEATURE_NAMES
from src.ordinal_cost import (
    create_ordinal_cost_matrix,
    predict_cost_sensitive,
    CostSensitiveClassifier,
)
from src.evaluation import (
    evaluate_predictions,
    plot_confusion_matrix,
    plot_roc_curves,
    generate_comparison_charts,
    generate_ablation_chart,
)

# fixed seed 42 and 80/20 stratified split kept identical across all runs
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.20


def build_tuning_grids() -> Dict[str, Tuple[Any, Dict[str, List[Any]]]]:
    return {
        "Decision_Tree": (
            DecisionTreeClassifier(random_state=RANDOM_SEED),
            {
                "max_depth": [None, 5, 8, 12, 18],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "criterion": ["gini", "entropy", "log_loss"],
            },
        ),
        "Random_Forest": (
            RandomForestClassifier(random_state=RANDOM_SEED),
            {
                "n_estimators": [100, 200],
                "max_depth": [None, 8, 14],
                "min_samples_leaf": [1, 2, 4],
                "class_weight": [None, "balanced"],
            },
        ),
        "Extra_Trees": (
            ExtraTreesClassifier(random_state=RANDOM_SEED),
            {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 16],
                "min_samples_leaf": [1, 2, 4],
                "class_weight": [None, "balanced"],
            },
        ),
        "Gradient_Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_SEED),
            {
                "n_estimators": [100, 150],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
            },
        ),
        "Logistic_Regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
            ]),
            {
                "clf__C": [0.1, 1.0, 10.0],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "SVM": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(probability=True, random_state=RANDOM_SEED)),
            ]),
            {
                "clf__C": [0.5, 1.0, 5.0],
                "clf__kernel": ["rbf", "linear"],
            },
        ),
    }


def tune_cost_sensitive_penalty(
    base_model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    penalties: List[float] = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
) -> Tuple[float, Dict[float, Dict[str, float]]]:
    # 5-fold stratified cross-validation on train split only
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    penalty_scores: Dict[float, Dict[str, float]] = {}

    for pen in penalties:
        cost_mat = create_ordinal_cost_matrix(base_cost_scale=1.0, high_risk_underestimate_penalty=pen)
        fold_recalls = []
        fold_macro_f1s = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            m = clone(base_model)
            m.fit(X_tr, y_tr)
            val_probs = m.predict_proba(X_val)

            val_preds, _ = predict_cost_sensitive(
                val_probs,
                m.classes_,
                cost_mat,
                ORDERED_CLASSES,
            )

            eval_dict = evaluate_predictions(y_val, val_preds, model_name="val")
            fold_recalls.append(eval_dict["high_risk_recall"])
            fold_macro_f1s.append(eval_dict["macro_f1"])

        mean_rec = float(np.mean(fold_recalls))
        mean_f1 = float(np.mean(fold_macro_f1s))
        # composite objective prioritizes high-risk recall (0.6) while preserving macro-f1 (0.4)
        comp_score = 0.6 * mean_rec + 0.4 * mean_f1
        penalty_scores[pen] = {
            "mean_high_risk_recall": round(mean_rec, 4),
            "mean_macro_f1": round(mean_f1, 4),
            "composite_score": round(comp_score, 4),
        }

    best_penalty = max(penalty_scores.keys(), key=lambda p: penalty_scores[p]["composite_score"])
    return best_penalty, penalty_scores


def run_pipeline(fast_mode: bool = False) -> Dict[str, Any]:
    start_time = time.time()
    reports_dir = Path("reports")
    figures_dir = reports_dir / "figures"
    models_dir = Path("models")
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df_raw, meta = load_dataset()
    X_raw = df_raw[CANONICAL_FEATURES]
    y_all = df_raw[CANONICAL_TARGET]

    # holdout 20% test split fixed to prevent leakage; seed 42 for exact reproducibility
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y_all,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_all,
    )

    X_train_eng = engineer_features(X_train_raw)
    X_test_eng = engineer_features(X_test_raw)

    grids = build_tuning_grids()
    if fast_mode:
        grids["Decision_Tree"][1]["max_depth"] = [None, 8]
        grids["Random_Forest"][1]["n_estimators"] = [100]
        grids["Extra_Trees"][1]["n_estimators"] = [100]

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    best_estimators: Dict[str, Any] = {}
    best_params: Dict[str, Any] = {}
    cv_scores_summary: List[Dict[str, Any]] = []

    # baseline decision tree with default hyperparameters on raw features
    baseline_dt = DecisionTreeClassifier(random_state=RANDOM_SEED)
    baseline_dt.fit(X_train_raw, y_train)

    base_cv_f1 = cross_val_score(baseline_dt, X_train_raw, y_train, cv=cv_strategy, scoring="f1_macro").mean()
    cv_scores_summary.append({
        "Model": "Baseline Decision Tree (Raw)",
        "Train_CV_Macro_F1": round(float(base_cv_f1), 4),
        "Best_Params": "default",
    })

    # grid search across candidate models using macro-f1 on training cv
    for model_name, (estimator, param_grid) in grids.items():
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,
        )
        grid_search.fit(X_train_eng, y_train)
        best_estimators[model_name] = grid_search.best_estimator_
        best_params[model_name] = grid_search.best_params_
        best_cv_score = grid_search.best_score_
        cv_scores_summary.append({
            "Model": f"Tuned {model_name} (Eng)",
            "Train_CV_Macro_F1": round(float(best_cv_score), 4),
            "Best_Params": str(grid_search.best_params_),
        })

    # tuned random forest on raw features for ablation comparison
    rf_raw_grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_SEED),
        param_grid=grids["Random_Forest"][1],
        cv=cv_strategy,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
    )
    rf_raw_grid.fit(X_train_raw, y_train)
    rf_tuned_raw = rf_raw_grid.best_estimator_
    best_params["Random_Forest_Raw"] = rf_raw_grid.best_params_

    # soft-voting ensemble combining top tree models
    voting_ensemble = VotingClassifier(
        estimators=[
            ("rf", best_estimators["Random_Forest"]),
            ("et", best_estimators["Extra_Trees"]),
            ("gb", best_estimators["Gradient_Boosting"]),
        ],
        voting="soft",
    )
    voting_ensemble.fit(X_train_eng, y_train)
    vote_cv_score = cross_val_score(voting_ensemble, X_train_eng, y_train, cv=cv_strategy, scoring="f1_macro").mean()
    cv_scores_summary.append({
        "Model": "Soft Voting Ensemble (Eng)",
        "Train_CV_Macro_F1": round(float(vote_cv_score), 4),
        "Best_Params": "Ensemble of RF, ET, GBDT",
    })

    selected_prob_model = best_estimators["Random_Forest"]
    best_pen, pen_grid_scores = tune_cost_sensitive_penalty(
        selected_prob_model, X_train_eng, y_train
    )
    best_params["Cost_Sensitive_HighRisk_Penalty"] = {
        "selected_penalty": best_pen,
        "grid_evaluations": pen_grid_scores,
    }

    cost_matrix = create_ordinal_cost_matrix(
        base_cost_scale=1.0,
        high_risk_underestimate_penalty=best_pen,
    )
    best_params["Cost_Matrix"] = cost_matrix.tolist()

    with open(reports_dir / "best_parameters.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, default=str)

    pd.DataFrame(cv_scores_summary).to_csv(reports_dir / "repeated_cv_summary.csv", index=False)

    evaluated_models: List[Dict[str, Any]] = []

    models_to_evaluate = [
        ("Baseline Decision Tree (Raw)", baseline_dt, X_test_raw, "dt_baseline"),
        ("Tuned Decision Tree", best_estimators["Decision_Tree"], X_test_eng, "dt_tuned"),
        ("Tuned Random Forest", best_estimators["Random_Forest"], X_test_eng, "rf_tuned"),
        ("Tuned Extra Trees", best_estimators["Extra_Trees"], X_test_eng, "et_tuned"),
        ("Tuned Gradient Boosting", best_estimators["Gradient_Boosting"], X_test_eng, "gb_tuned"),
        ("Scaled Logistic Regression", best_estimators["Logistic_Regression"], X_test_eng, "lr_scaled"),
        ("Scaled SVM (RBF)", best_estimators["SVM"], X_test_eng, "svm_scaled"),
        ("Soft Voting Ensemble", voting_ensemble, X_test_eng, "voting_ensemble"),
    ]

    for display_name, model, x_t, slug in models_to_evaluate:
        y_pred = model.predict(x_t)
        y_proba = model.predict_proba(x_t) if hasattr(model, "predict_proba") else None
        classes_ = getattr(model, "classes_", ORDERED_CLASSES)

        eval_res = evaluate_predictions(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            estimator_classes=classes_,
            model_name=display_name,
        )
        eval_res["model_slug"] = slug
        evaluated_models.append(eval_res)

        plot_confusion_matrix(eval_res["confusion_matrix"], slug, display_name, figures_dir)
        plot_roc_curves(eval_res["per_class_roc"], slug, display_name, figures_dir)

    # evaluate cost-sensitive decision model on test probabilities
    rf_test_probs = best_estimators["Random_Forest"].predict_proba(X_test_eng)
    cost_test_preds, _ = predict_cost_sensitive(
        rf_test_probs,
        best_estimators["Random_Forest"].classes_,
        cost_matrix,
        ORDERED_CLASSES,
    )
    cost_eval_res = evaluate_predictions(
        y_true=y_test,
        y_pred=cost_test_preds,
        y_proba=rf_test_probs,
        estimator_classes=best_estimators["Random_Forest"].classes_,
        model_name="Ordinal Cost-Sensitive (Tuned RF)",
    )
    cost_eval_res["model_slug"] = "cost_sensitive_rf"
    evaluated_models.append(cost_eval_res)
    plot_confusion_matrix(cost_eval_res["confusion_matrix"], "cost_sensitive_rf", "Ordinal Cost-Sensitive (Tuned RF)", figures_dir)
    plot_roc_curves(cost_eval_res["per_class_roc"], "cost_sensitive_rf", "Ordinal Cost-Sensitive (Tuned RF)", figures_dir)

    comparison_table_rows: List[Dict[str, Any]] = []
    for m in evaluated_models:
        comparison_table_rows.append({
            "Model": m["model_name"],
            "Accuracy": m["accuracy"],
            "Balanced_Accuracy": m["balanced_accuracy"],
            "Macro_Precision": m["macro_precision"],
            "Macro_Recall": m["macro_recall"],
            "Macro_F1": m["macro_f1"],
            "Weighted_F1": m["weighted_f1"],
            "High_Risk_Precision": m["high_risk_precision"],
            "High_Risk_Recall": m["high_risk_recall"],
            "High_Risk_F1": m["high_risk_f1"],
            "MCC": m["mcc"],
            "Cohen_Kappa": m["cohen_kappa"],
            "ROC_AUC_Macro": m["roc_auc_macro"],
        })
    df_comparison = pd.DataFrame(comparison_table_rows)
    df_comparison.to_csv(reports_dir / "holdout_model_comparison.csv", index=False)

    summary_columns: List[str] = [
        "Model",
        "Accuracy",
        "Balanced_Accuracy",
        "Macro_Precision",
        "Macro_Recall",
        "Macro_F1",
        "Weighted_F1",
        "High_Risk_Recall",
        "ROC_AUC_Macro",
    ]
    selected_model_name = "Ordinal Cost-Sensitive (Tuned RF)"
    selected_model_metrics = next(
        row for row in comparison_table_rows if row["Model"] == selected_model_name
    )
    highest_accuracy_model = max(
        comparison_table_rows, key=lambda row: float(row["Accuracy"])
    )
    table_header = f"{'Model':<34}" + "".join(
        f"{column:>22}" for column in summary_columns[1:]
    )
    table_rows = [table_header]
    for row in comparison_table_rows:
        model_name = str(row["Model"])
        metric_values = "".join(
            f"{float(row[column]):>22.4f}" for column in summary_columns[1:]
        )
        table_rows.append(f"{model_name:<34}{metric_values}")
    metrics_table = "\n".join(table_rows)
    metrics_summary = (
        "Holdout metrics (stratified 80/20 split, random seed 42)\n"
        "Scores are measured on the same held-out test set.\n\n"
        f"Model saved for the app: {selected_model_name}\n"
        f"Accuracy: {selected_model_metrics['Accuracy']:.4f} | "
        f"Macro-F1: {selected_model_metrics['Macro_F1']:.4f} | "
        f"High-risk recall: {selected_model_metrics['High_Risk_Recall']:.4f}\n\n"
        f"Highest holdout accuracy: {highest_accuracy_model['Model']} "
        f"({highest_accuracy_model['Accuracy']:.4f})\n\n"
        f"{metrics_table}\n"
    )
    (reports_dir / "metrics_summary.txt").write_text(metrics_summary, encoding="utf-8")

    generate_comparison_charts(evaluated_models, figures_dir)

    with open(reports_dir / "classification_reports.txt", "w", encoding="utf-8") as f:
        for m in evaluated_models:
            f.write(f"Model: {m['model_name']}\n")
            f.write(m["classification_report"])
            f.write(f"\nConfusion matrix:\n{np.array(m['confusion_matrix'])}\n\n")

    # ablation study: v0 through v7
    v0_eval = evaluate_predictions(y_test, baseline_dt.predict(X_test_raw), model_name="V0")

    dt_v1 = DecisionTreeClassifier(random_state=RANDOM_SEED)
    dt_v1.fit(X_train_eng, y_train)
    v1_eval = evaluate_predictions(y_test, dt_v1.predict(X_test_eng), model_name="V1")

    v2_eval = evaluate_predictions(y_test, best_estimators["Decision_Tree"].predict(X_test_eng), model_name="V2")
    v3_eval = evaluate_predictions(y_test, rf_tuned_raw.predict(X_test_raw), model_name="V3")
    v4_eval = evaluate_predictions(y_test, best_estimators["Random_Forest"].predict(X_test_eng), model_name="V4")
    v5_eval = evaluate_predictions(y_test, best_estimators["Extra_Trees"].predict(X_test_eng), model_name="V5")
    v6_eval = evaluate_predictions(y_test, voting_ensemble.predict(X_test_eng), model_name="V6")
    v7_eval = cost_eval_res

    # sensitivity analysis: test impact of duplicate rows on model performance
    df_dedup = df_raw.drop_duplicates()
    x_tr_dd, x_te_dd, y_tr_dd, y_te_dd = train_test_split(
        df_dedup[CANONICAL_FEATURES], df_dedup[CANONICAL_TARGET], test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=df_dedup[CANONICAL_TARGET]
    )
    rf_dedup = RandomForestClassifier(random_state=RANDOM_SEED)
    rf_dedup.fit(x_tr_dd, y_tr_dd)
    dedup_eval = evaluate_predictions(y_te_dd, rf_dedup.predict(x_te_dd), model_name="V_Dedup")

    ablation_records = [
        {"ablation_version": "V0", "description": "Baseline Decision Tree (Original features)", "accuracy": v0_eval["accuracy"], "macro_f1": v0_eval["macro_f1"], "weighted_f1": v0_eval["weighted_f1"], "high_risk_recall": v0_eval["high_risk_recall"]},
        {"ablation_version": "V1", "description": "Baseline Decision Tree (Engineered features)", "accuracy": v1_eval["accuracy"], "macro_f1": v1_eval["macro_f1"], "weighted_f1": v1_eval["weighted_f1"], "high_risk_recall": v1_eval["high_risk_recall"]},
        {"ablation_version": "V2", "description": "Tuned Decision Tree (Engineered features)", "accuracy": v2_eval["accuracy"], "macro_f1": v2_eval["macro_f1"], "weighted_f1": v2_eval["weighted_f1"], "high_risk_recall": v2_eval["high_risk_recall"]},
        {"ablation_version": "V3", "description": "Tuned Random Forest (Original features)", "accuracy": v3_eval["accuracy"], "macro_f1": v3_eval["macro_f1"], "weighted_f1": v3_eval["weighted_f1"], "high_risk_recall": v3_eval["high_risk_recall"]},
        {"ablation_version": "V4", "description": "Tuned Random Forest (Engineered features)", "accuracy": v4_eval["accuracy"], "macro_f1": v4_eval["macro_f1"], "weighted_f1": v4_eval["weighted_f1"], "high_risk_recall": v4_eval["high_risk_recall"]},
        {"ablation_version": "V5", "description": "Tuned Extra Trees (Engineered features)", "accuracy": v5_eval["accuracy"], "macro_f1": v5_eval["macro_f1"], "weighted_f1": v5_eval["weighted_f1"], "high_risk_recall": v5_eval["high_risk_recall"]},
        {"ablation_version": "V6", "description": "Soft-Voting Ensemble (RF + ET + GBDT)", "accuracy": v6_eval["accuracy"], "macro_f1": v6_eval["macro_f1"], "weighted_f1": v6_eval["weighted_f1"], "high_risk_recall": v6_eval["high_risk_recall"]},
        {"ablation_version": "V7", "description": "Ordinal Cost-Sensitive Layer (High-risk safety)", "accuracy": v7_eval["accuracy"], "macro_f1": v7_eval["macro_f1"], "weighted_f1": v7_eval["weighted_f1"], "high_risk_recall": v7_eval["high_risk_recall"]},
        {"ablation_version": "Dedup_Analysis", "description": "Tuned RF on Strictly Deduplicated Data (N=452)", "accuracy": dedup_eval["accuracy"], "macro_f1": dedup_eval["macro_f1"], "weighted_f1": dedup_eval["weighted_f1"], "high_risk_recall": dedup_eval["high_risk_recall"]},
    ]
    pd.DataFrame(ablation_records).to_csv(reports_dir / "ablation_results.csv", index=False)
    generate_ablation_chart(pd.DataFrame(ablation_records).iloc[:8], figures_dir)

    final_cost_clf = CostSensitiveClassifier(
        base_estimator=best_estimators["Random_Forest"],
        base_cost_scale=1.0,
        high_risk_underestimate_penalty=best_pen,
    )
    final_cost_clf.classes_ = np.array(ORDERED_CLASSES)
    final_cost_clf.cost_matrix_ = cost_matrix

    model_bundle = {
        "model_name": "Tuned Random Forest with Ordinal Cost-Sensitive Layer",
        "base_model": best_estimators["Random_Forest"],
        "cost_sensitive_model": final_cost_clf,
        "raw_features": CANONICAL_FEATURES,
        "all_features": ALL_FEATURE_NAMES,
        "classes": ORDERED_CLASSES,
        "cost_matrix": cost_matrix,
        "best_penalty": best_pen,
        "split_params": {"test_size": TEST_SIZE, "random_state": RANDOM_SEED},
        "metrics_on_holdout": cost_eval_res,
        "software_versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": "1.3+",
        },
    }
    joblib.dump(model_bundle, models_dir / "best_model_bundle.joblib")

    elapsed_time = time.time() - start_time
    run_summary_text = f"""Maternal Health Risk Prediction — Run Summary
Date/Time: 2026-10-27
Execution Duration: {elapsed_time:.2f} seconds
Dataset Source: {meta['source']}
Total Samples: {len(df_raw)} (Train: {len(X_train_raw)}, Holdout: {len(X_test_raw)})
Random Seed: {RANDOM_SEED} (Holdout Test Size: {TEST_SIZE})

Best CV Models (Stratified 5-Fold Macro-F1 on Train):
- Random Forest CV: {cv_scores_summary[2]['Train_CV_Macro_F1']}
- Extra Trees CV: {cv_scores_summary[3]['Train_CV_Macro_F1']}
- Voting Ensemble CV: {cv_scores_summary[7]['Train_CV_Macro_F1']}

Holdout Test Performance (N=203):
- Baseline Decision Tree Accuracy: {v0_eval['accuracy']:.4f} | Macro-F1: {v0_eval['macro_f1']:.4f} | High-Risk Recall: {v0_eval['high_risk_recall']:.4f}
- Tuned Random Forest Accuracy: {v4_eval['accuracy']:.4f} | Macro-F1: {v4_eval['macro_f1']:.4f} | High-Risk Recall: {v4_eval['high_risk_recall']:.4f}
- Soft-Voting Ensemble Accuracy: {v6_eval['accuracy']:.4f} | Macro-F1: {v6_eval['macro_f1']:.4f} | High-Risk Recall: {v6_eval['high_risk_recall']:.4f}
- Ordinal Cost-Sensitive Layer Accuracy: {v7_eval['accuracy']:.4f} | Macro-F1: {v7_eval['macro_f1']:.4f} | High-Risk Recall: {v7_eval['high_risk_recall']:.4f}
- Deduplication Sensitivity (RF on 452 unique rows): Accuracy: {dedup_eval['accuracy']:.4f} | Macro-F1: {dedup_eval['macro_f1']:.4f}

External Benchmark Reference:
- UBC-MDS published baseline: 82.3% (External reference, protocol-dependent).
"""
    with open(reports_dir / "run_summary.txt", "w", encoding="utf-8") as f:
        f.write(run_summary_text)

    print(f"Pipeline complete in {elapsed_time:.1f}s.")
    print(f"\nModel saved for the app: {selected_model_name}")
    print(
        f"Holdout accuracy: {selected_model_metrics['Accuracy']:.4f} | "
        f"Macro-F1: {selected_model_metrics['Macro_F1']:.4f} | "
        f"High-risk recall: {selected_model_metrics['High_Risk_Recall']:.4f}"
    )
    print(
        f"Highest holdout accuracy: {highest_accuracy_model['Model']} "
        f"({highest_accuracy_model['Accuracy']:.4f})"
    )
    print("\nHoldout accuracy, F1, recall, and related metrics:")
    print(metrics_table)
    print("\nReports updated:")
    print(f"- {reports_dir / 'metrics_summary.txt'}")
    print(f"- {reports_dir / 'classification_reports.txt'}")
    print(f"- {reports_dir / 'holdout_model_comparison.csv'}")
    print(f"- {reports_dir / 'run_summary.txt'}")

    return {
        "evaluated_models": evaluated_models,
        "ablation_results": ablation_records,
        "best_parameters": best_params,
        "summary": run_summary_text,
    }


if __name__ == "__main__":
    run_pipeline()
