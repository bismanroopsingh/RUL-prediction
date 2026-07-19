import os
import joblib
import pandas as pd
import xgboost as xgb

from CMaps.config import (
    TEST_PATH,
    RUL_PATH,
    MODEL_DIR,
    OUTPUT_DIR,
)

from src.data_loader import (
    load_test_data,
    load_rul,
)

from src.feature_engineering import FeatureEngineer
from src.feature_selection import FeatureSelector
from src.evaluate import Evaluator


def main():

    print("=" * 60)
    print("NASA C-MAPSS Prediction")
    print("=" * 60)

    ############################################################
    # Load Test Data
    ############################################################

    print("Loading NASA Test Dataset...")

    test = load_test_data(TEST_PATH)

    print(test.head())
    print()

    print(f"Dataset Shape : {test.shape}")
    print()

    ############################################################
    # Save Engine IDs
    ############################################################

    engine_ids = test["engine_id"].copy()

    ############################################################
    # Feature Engineering
    ############################################################

    print("=" * 60)
    print("Feature Engineering")
    print("=" * 60)

    engineer = FeatureEngineer()

    test = engineer.engineer(test)

    ############################################################
    # Separate Features
    ############################################################

    X_test = test.drop(
        columns=["engine_id", "cycle"],
        errors="ignore",
    )

    ############################################################
    # Feature Selection
    ############################################################

    print("=" * 60)
    print("Feature Selection")
    print("=" * 60)

    selector = FeatureSelector()

    selector.load_features()

    X_test = selector.transform(X_test)

    print(f"Selected Features : {X_test.shape[1]}")
    print()

    ############################################################
    # Scaling
    ############################################################

    print("=" * 60)
    print("Scaling Features")
    print("=" * 60)

    engineer.load_scaler()

    X_test = pd.DataFrame(

        engineer.transform(X_test),

        columns=X_test.columns,

        index=X_test.index,

    )

    print("Scaling Complete")
    print()

    ############################################################
    # Load Model
    ############################################################

    print("=" * 60)
    print("Loading Trained Model")
    print("=" * 60)

    model = joblib.load(
        MODEL_DIR / "xgboost.pkl"
    )

    ############################################################
    # Prediction
    ############################################################

    print("=" * 60)
    print("Predicting Remaining Useful Life")
    print("=" * 60)

    predictions = model.predict(X_test)

    prediction_df = pd.DataFrame({

        "engine_id": engine_ids,

        "Predicted_RUL": predictions,

    })

    ############################################################
    # Keep Last Cycle Prediction
    ############################################################

    final_prediction = (

        prediction_df

        .groupby("engine_id")

        .last()

        .reset_index()

    )

    ############################################################
    # Save Predictions
    ############################################################

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    prediction_df.to_csv(

        OUTPUT_DIR / "all_cycle_predictions.csv",

        index=False,

    )

    final_prediction.to_csv(

        OUTPUT_DIR / "final_engine_predictions.csv",

        index=False,

    )

    print(
        "Predictions saved."
    )

    ############################################################
    # Evaluation
    ############################################################

    print("=" * 60)
    print("Evaluating Predictions")
    print("=" * 60)

    rul = load_rul(RUL_PATH)

    evaluator = Evaluator()

    metrics = evaluator.evaluate(

        rul["RUL"],

        final_prediction["Predicted_RUL"],

    )

    ############################################################
    # Display Sample
    ############################################################

    result = pd.DataFrame({

        "Engine":

        final_prediction["engine_id"],

        "Actual RUL":

        rul["RUL"],

        "Predicted RUL":

        final_prediction["Predicted_RUL"],

    })

    print()

    print("=" * 60)
    print("Sample Predictions")
    print("=" * 60)

    print(result.head(20))

    ############################################################
    # Save Final Results
    ############################################################

    result.to_csv(

        OUTPUT_DIR / "prediction_results.csv",

        index=False,

    )

    print()

    print("=" * 60)
    print("Prediction Complete")
    print("=" * 60)

    print(f"MAE        : {metrics['MAE']:.4f}")
    print(f"RMSE       : {metrics['RMSE']:.4f}")
    print(f"R2         : {metrics['R2']:.4f}")
    print(f"NASA Score : {metrics['NASA Score']:.4f}")

    print()

    print(f"Results saved to : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()