#!/usr/bin/env python3
"""Train CatBoost model with CV and export artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_accident_risk.config import DROP_COLUMNS, ID_COL, TARGET_COL, get_catboost_params
from traffic_accident_risk.evaluation import evaluate_classifier, print_cv_summary, summarize_feature_importance
from traffic_accident_risk.preprocessing import preprocess_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CatBoost model for binary classification.")
    parser.add_argument("--train-path", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument("--test-path", type=Path, default=ROOT / "data" / "test.csv")
    parser.add_argument(
        "--sample-submission-path",
        type=Path,
        default=ROOT / "data" / "sample_submission.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "model_b")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--task-type", type=str, choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def save_submission(
    sample_submission_path: Path,
    predictions: np.ndarray,
    output_path: Path,
) -> None:
    if sample_submission_path.exists():
        submission = pd.read_csv(sample_submission_path)
        target_cols = [col for col in submission.columns if col.lower() != "id"]
        target_col = target_cols[0] if target_cols else submission.columns[-1]
        submission[target_col] = predictions
    else:
        submission = pd.DataFrame({"prediction": predictions})
    submission.to_csv(output_path, index=False)
    print(f"submission saved: {output_path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "fold_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    train_data = pd.read_csv(args.train_path).drop(columns=[ID_COL], errors="ignore")
    test_data = pd.read_csv(args.test_path).drop(columns=[ID_COL], errors="ignore")

    train_data = preprocess_dataframe(train_data, drop_list=DROP_COLUMNS, is_train=True, verbose=False)
    test_data = preprocess_dataframe(test_data, drop_list=DROP_COLUMNS, is_train=False, verbose=False)

    feature_cols = [col for col in train_data.columns if col != TARGET_COL]
    cat_features = feature_cols

    oof_preds = np.zeros(len(train_data), dtype=float)
    test_preds = np.zeros(len(test_data), dtype=float)
    fold_importances: list[np.ndarray] = []
    metrics_history: dict[str, list[float]] = defaultdict(list)

    kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    params = get_catboost_params(task_type=args.task_type, random_state=args.random_state)

    for fold, (train_idx, val_idx) in enumerate(
        kf.split(train_data, train_data[TARGET_COL]), start=1
    ):
        print(f"\n============= Training Fold {fold}/{args.n_splits} =============")
        fold_train = train_data.iloc[train_idx]
        fold_val = train_data.iloc[val_idx]

        x_train = fold_train[feature_cols]
        y_train = fold_train[TARGET_COL].astype(int)
        x_val = fold_val[feature_cols]
        y_val = fold_val[TARGET_COL].astype(int)

        train_pool = Pool(x_train, label=y_train, cat_features=cat_features)
        eval_pool = Pool(x_val, label=y_val, cat_features=cat_features)

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=eval_pool, verbose=100, early_stopping_rounds=250)

        model_path = model_dir / f"catboost_fold_{fold}.cbm"
        model.save_model(str(model_path))
        print(f"model saved: {model_path}")

        val_prob = model.predict_proba(x_val)[:, 1]
        val_pred = (val_prob > args.threshold).astype(int)
        fold_metrics = evaluate_classifier(y_val, val_pred, val_prob)
        for metric_name, metric_value in fold_metrics.items():
            metrics_history[metric_name].append(metric_value)

        oof_preds[val_idx] = val_prob
        test_preds += model.predict_proba(test_data[feature_cols])[:, 1] / args.n_splits
        fold_importances.append(model.get_feature_importance())

    print_cv_summary(metrics_history)

    ranking, mean_importance = summarize_feature_importance(
        feature_names=feature_cols,
        fold_importances=fold_importances,
        output_csv=args.output_dir / "feature_importance.csv",
        output_png=args.output_dir / "feature_importance_top50.png",
        top_n=50,
    )
    print(f"mean feature importance: {mean_importance:.6f}")
    print(f"top feature: {ranking.iloc[0]['feature']}")

    save_submission(
        sample_submission_path=args.sample_submission_path,
        predictions=test_preds,
        output_path=args.output_dir / "submission.csv",
    )

    pd.DataFrame({"oof_pred": oof_preds}).to_csv(args.output_dir / "oof_predictions.csv", index=False)
    with open(args.output_dir / "cv_metrics.json", "w", encoding="utf-8") as fp:
        json.dump(metrics_history, fp, ensure_ascii=False, indent=2)
    print(f"artifacts saved in: {args.output_dir}")


if __name__ == "__main__":
    main()

