# ==========================================================
# NASA C-MAPSS Remaining Useful Life Prediction
# Complete Training Pipeline
# ==========================================================

import os
import json
import joblib
import warnings
import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from src.evaluate import (
    Evaluator,
)
from sklearn.model_selection import GroupShuffleSplit

# ----------------------------------------------------------
# Ignore warnings
# ----------------------------------------------------------

warnings.filterwarnings("ignore")

# ----------------------------------------------------------
# Project Configuration
# ----------------------------------------------------------

from CMaps.config import *

# ----------------------------------------------------------
# Data Loading
# ----------------------------------------------------------

from src.data_loader import (
    load_training_data,
)

from src.preprocessing import (
    add_rul,
)

# ----------------------------------------------------------
# Feature Engineering
# ----------------------------------------------------------

from src.feature_engineering import (
    FeatureEngineer,
)

# ----------------------------------------------------------
# Feature Selection
# ----------------------------------------------------------

from src.feature_selection import (
    FeatureSelector,
)

# ----------------------------------------------------------
# Benchmarking
# ----------------------------------------------------------

from src.model_selection import (
    ModelBenchmark,
)

# ----------------------------------------------------------
# Hyperparameter Tuning
# ----------------------------------------------------------

from src.tune import (
    XGBoostTuner,
)

# ----------------------------------------------------------
# Evaluation
# ----------------------------------------------------------



# ==========================================================
# Create folders
# ==========================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ==========================================================
# MLflow
# ==========================================================

mlflow.set_experiment(
    "NASA_RUL_XGBoost"
)

with mlflow.start_run():

    # ==========================================================
    # STEP 1
    # Load Dataset
    # ==========================================================

    print("=" * 60)
    print("Loading NASA Training Dataset...")
    print("=" * 60)

    df = load_training_data(
        TRAIN_PATH
    )

    print(df.head())

    print()

    print(f"Dataset Shape : {df.shape}")

    print()

    # ==========================================================
    # STEP 2
    # Generate RUL
    # ==========================================================

    print("=" * 60)
    print("Generating Remaining Useful Life")
    print("=" * 60)

    df = add_rul(df)

    print(df[
        [
            "engine_id",
            "cycle",
            "RUL"
        ]
    ].head())

    print()

    # ==========================================================
    # STEP 3
    # Feature Engineering
    # ==========================================================

    print("=" * 60)
    print("Feature Engineering")
    print("=" * 60)

    engineer = FeatureEngineer()

    df = engineer.engineer(df)

    print("Feature Engineering Completed")

    print(f"Dataset Shape : {df.shape}")

    print()

    # ==========================================================
    # STEP 4
    # Split Features & Target
    # ==========================================================

    groups = df["engine_id"]

    y = df["RUL"]

    drop_columns = [
        "engine_id",
        "cycle",
        "RUL",
    ]

    X = df.drop(
        columns=drop_columns
    )

    print(f"Total Features : {X.shape[1]}")

    print()

    # ==========================================================
    # STEP 5
    # Feature Selection
    # ==========================================================

    print("=" * 60)
    print("Feature Selection")
    print("=" * 60)

    selector = FeatureSelector(
        variance_threshold=0.0,
        correlation_threshold=0.95,
        top_features=60,
    )

    X = selector.fit_transform(
        X,
        y,
    )

    print(f"Remaining Features : {X.shape[1]}")
    print()

    # ==========================================================
    # STEP 7
    # Train / Validation Split
    # ==========================================================

    print("=" * 60)
    print("Creating Group-wise Train/Validation Split")
    print("=" * 60)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_idx, valid_idx = next(
        splitter.split(
            X,
            y,
            groups,
        )
    )

    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]

    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    groups_train = groups.iloc[train_idx]
    groups_valid = groups.iloc[valid_idx]

    print(f"Training Samples   : {len(X_train)}")
    print(f"Validation Samples : {len(X_valid)}")
    print()

    print(
        f"Training Engines : {groups_train.nunique()}"
    )

    print(
        f"Validation Engines : {groups_valid.nunique()}"
    )

    print()

    
        # ==========================================================
    # STEP 6
    # Scale Features
    # ==========================================================

    print("=" * 60)
    print("Scaling Features")
    print("=" * 60)

    engineer.fit_scaler(X_train)

    X_train = pd.DataFrame(
        engineer.transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    X_valid = pd.DataFrame(
        engineer.transform(X_valid),
        columns=X_valid.columns,
        index=X_valid.index,
    )

    print("Scaling completed.")
    print()

 
    # ==========================================================
    # Save Processed Dataset
    # ==========================================================

    X_train.to_csv(
        PROCESSED_DATA / "X_train.csv",
        index=False,
    )

    X_valid.to_csv(
        PROCESSED_DATA / "X_valid.csv",
        index=False,
    )

    pd.DataFrame(
        y_train,
        columns=["RUL"]
    ).to_csv(
        PROCESSED_DATA / "y_train.csv",
        index=False,
    )

    pd.DataFrame(
        y_valid,
        columns=["RUL"]
    ).to_csv(
        PROCESSED_DATA / "y_valid.csv",
        index=False,
    )

    print("Processed datasets saved.")
    print()

    # ==========================================================
    # STEP 8
    # Benchmark Models
    # ==========================================================

    print("=" * 60)
    print("Benchmarking Models")
    print("=" * 60)

    benchmark = ModelBenchmark()

    results = benchmark.evaluate(
        X_train,
        X_valid,
        y_train,
        y_valid,
    )

    print(results)

    results.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
    )

    print()

    print("Benchmark completed.")
    print()

    # ==========================================================
    # STEP 9
    # Hyperparameter Optimization
    # ==========================================================

    print("=" * 60)
    print("Optuna Hyperparameter Optimization")
    print("=" * 60)

    tuner = XGBoostTuner(
        X_train,
        y_train,
        groups_train,
        n_splits=5,
        random_state=RANDOM_STATE,
    )

    study = tuner.optimize(
        n_trials=50,      # Reduce to 20 while testing if needed
    )

    best_params = study.best_params
    mlflow.log_params(best_params)

    print("\nBest Parameters:")
    for key, value in best_params.items():
        print(f"{key}: {value}")

    print()

    # Save best parameters and optimization history
    tuner.save_best_params()
    tuner.save_history()

    # ==========================================================
    # STEP 10
    # Train Final XGBoost Model
    # ==========================================================

    print("=" * 60)
    print("Training Final XGBoost Model")
    print("=" * 60)

    model = tuner.get_best_model()


    print("Final model trained successfully.")
    # ==========================================================
    # Evaluate Final Model
    # ==========================================================

    from src.evaluate import Evaluator

    print("=" * 60)
    print("Evaluating Final Model")
    print("=" * 60)

    pred = model.predict(X_valid)

    evaluator = Evaluator()

    metrics = evaluator.evaluate(
        y_valid,
        pred,
    )
    importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    })

    importance.sort_values(
        "Importance",
        ascending=False,
        inplace=True
    )

    importance.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False,
    )
    mlflow.log_metric("MAE", metrics["MAE"])
    mlflow.log_metric("RMSE", metrics["RMSE"])
    mlflow.log_metric("NASA Score", metrics["NASA Score"])
    mlflow.log_metric("R2", metrics["R2"])
    mlflow.log_artifact(
    OUTPUT_DIR / "metrics.json"
    )

    mlflow.log_artifact(
        OUTPUT_DIR / "feature_importance.csv"
    )
    joblib.dump(
        model,
        MODEL_DIR / "xgboost.pkl",
    )

    mlflow.xgboost.log_model(
        xgb_model=model,
        name="model"
    )
    training_summary = {
    "Best RMSE (Optuna CV)": study.best_value,
    "Validation MAE": metrics["MAE"],
    "Validation RMSE": metrics["RMSE"],
    "Validation R2": metrics["R2"],
    "NASA Score": metrics["NASA Score"],
    "Number of Features": len(X_train.columns),
    "Best Parameters": best_params,
}

    with open(
        OUTPUT_DIR / "training_summary.json",
        "w"
    ) as f:
        json.dump(training_summary, f, indent=4)
    mlflow.log_artifact(
        OUTPUT_DIR / "training_summary.json"
    )
    mlflow.log_metric(
    "Optuna_CV_RMSE",
    study.best_value
)
    mlflow.log_metric(
    "Selected_Features",
    len(X_train.columns)
    )
    print(f"Model saved to {MODEL_DIR / 'xgboost.pkl'}")
    print(f"Feature importance saved to {OUTPUT_DIR / 'feature_importance.csv'}")
    print(f"Training summary saved to {OUTPUT_DIR / 'training_summary.json'}")
    print(f"Model saved to {MODEL_DIR / 'xgboost.pkl'}")
    print()