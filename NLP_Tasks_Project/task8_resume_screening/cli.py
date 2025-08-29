from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .data import load_dataset, train_val_test_split, coerce_score_and_clip
from .preprocess import add_skill_overlap_features
from .models import TfidfRidgeConfig, build_tfidf_ridge_regressor, fit_tfidf_ridge, predict
from .evaluate import regression_metrics, to_class


def cmd_train(args):
    df = load_dataset(args.csv)
    df = coerce_score_and_clip(df)
    df = add_skill_overlap_features(df)

    train_df, val_df, test_df = train_val_test_split(df, train_size=0.8, val_size=0.1)

    y_train = train_df["match_score"].to_numpy(dtype=float)
    y_val = val_df["match_score"].to_numpy(dtype=float)
    y_test = test_df["match_score"].to_numpy(dtype=float)

    config = TfidfRidgeConfig()
    pipe, params = build_tfidf_ridge_regressor(config)
    transformer, estimator = fit_tfidf_ridge(pipe, train_df, y_train)

    # Evaluate on validation
    val_preds = predict((transformer, estimator), val_df)
    val_report = regression_metrics(y_val, val_preds)

    print("Validation metrics:")
    print(json.dumps(val_report, indent=2))

    # Test
    test_preds = predict((transformer, estimator), test_df)
    test_report = regression_metrics(y_test, test_preds)
    print("\nTest metrics:")
    print(json.dumps(test_report, indent=2))

    # Save minimal model components
    os.makedirs(args.out_dir, exist_ok=True)
    import joblib
    joblib.dump(transformer, os.path.join(args.out_dir, "transformer.joblib"))
    joblib.dump(estimator, os.path.join(args.out_dir, "estimator.joblib"))
    print(f"\nSaved model artifacts to: {args.out_dir}")


def cmd_score(args):
    import joblib
    transformer = joblib.load(os.path.join(args.model_dir, "transformer.joblib"))
    estimator = joblib.load(os.path.join(args.model_dir, "estimator.joblib"))

    # Build a tiny DataFrame to reuse existing feature pipeline
    import pandas as pd
    df = pd.DataFrame([
        {
            "job_description": args.job,
            "resume": args.resume,
            "match_score": 0,
        }
    ])
    from .preprocess import add_skill_overlap_features
    df = add_skill_overlap_features(df)

    preds = predict((transformer, estimator), df)
    score = float(np.clip(preds[0], 1.0, 5.0))
    print(json.dumps({"predicted_match_score": score, "rounded_class": int(round(score))}, indent=2))


def build_parser():
    p = argparse.ArgumentParser(description="Task 8: Resume Screening using NLP")
    sub = p.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train TF-IDF Ridge baseline")
    p_train.add_argument("--csv", required=True, help="Path to resume_job_matching_dataset.csv")
    p_train.add_argument("--out_dir", default="NLP_Tasks_Project/task8_model", help="Output dir for model")
    p_train.set_defaults(func=cmd_train)

    p_score = sub.add_parser("score", help="Score a single JD/Resume pair")
    p_score.add_argument("--model_dir", required=True, help="Directory containing transformer.joblib and estimator.joblib")
    p_score.add_argument("--job", required=True, help="Job description text")
    p_score.add_argument("--resume", required=True, help="Resume text")
    p_score.set_defaults(func=cmd_score)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


