import os
import joblib
import xgboost as xgb
import shap
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    TRAIN_PATH,
    MODEL_DIR,
    OUTPUT_DIR,
)

from src.data_loader import load_training_data
from src.preprocessing import add_rul
from src.feature_engineering import FeatureEngineer
from src.feature_selection import FeatureSelector


def main():

    print("=" * 60)
    print("Generating SHAP Explanations")
    print("=" * 60)

    ############################################################
    # Load Training Data
    ############################################################

    print("Loading training data...")

    train = load_training_data(TRAIN_PATH)

    train = add_rul(train)

    ############################################################
    # Feature Engineering
    ############################################################

    print("Applying feature engineering...")

    engineer = FeatureEngineer()

    train = engineer.engineer(train)

    ############################################################
    # Prepare Features
    ############################################################

    X = train.drop(
        columns=[
            "engine_id",
            "cycle",
            "RUL",
        ],
        errors="ignore",
    )

    ############################################################
    # Feature Selection
    ############################################################

    print("Applying feature selection...")

    selector = FeatureSelector()

    selector.load_features()

    X = selector.transform(X)

    ############################################################
    # Scaling
    ############################################################

    print("Scaling features...")

    engineer.load_scaler()

    X = pd.DataFrame(
        engineer.transform(X),
        columns=X.columns,
        index=X.index,
    )

    ############################################################
    # Load Model
    ############################################################

    print("Loading trained model...")

    model = joblib.load(
        MODEL_DIR / "xgboost.pkl"
    )

    ############################################################
    # Native XGBoost SHAP
    ############################################################

    print("Computing SHAP values...")

    booster = model.get_booster()

    dmatrix = xgb.DMatrix(
        X,
        feature_names=X.columns.tolist(),
    )

    shap_values = booster.predict(
        dmatrix,
        pred_contribs=True,
    )

    # Remove bias term (last column)
    shap_values = shap_values[:, :-1]

    ############################################################
    # Create Output Directory
    ############################################################

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    ############################################################
    # SHAP Summary Plot
    ############################################################

    print("Saving SHAP summary plot...")

    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        shap_values,
        X,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_summary.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    ############################################################
    # SHAP Bar Plot
    ############################################################

    print("Saving SHAP bar plot...")

    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_bar.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    ############################################################
    # SHAP Importance CSV
    ############################################################

    importance = pd.DataFrame({

        "Feature": X.columns,

        "Mean_SHAP":

        abs(shap_values).mean(axis=0)

    })

    importance = importance.sort_values(
        "Mean_SHAP",
        ascending=False,
    )

    importance.to_csv(
        OUTPUT_DIR / "shap_importance.csv",
        index=False,
    )

    ############################################################
    # Top 5 Dependence Plots
    ############################################################

    print("Saving dependence plots...")

    top_features = importance.head(5)["Feature"]

    for feature in top_features:

        plt.figure(figsize=(8, 6))

        shap.dependence_plot(
            feature,
            shap_values,
            X,
            show=False,
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"{feature}_dependence.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    ############################################################
    # Display Top Features
    ############################################################

    print()

    print("=" * 60)
    print("Top 15 Important Features")
    print("=" * 60)

    print(importance.head(15))

    print()

    print("=" * 60)
    print("SHAP Analysis Complete")
    print("=" * 60)

    print("Generated Files:")
    print()

    print("✔ shap_summary.png")
    print("✔ shap_bar.png")
    print("✔ shap_importance.csv")

    for feature in top_features:
        print(f"✔ {feature}_dependence.png")

    print()
    print(f"Saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()