import joblib
import pandas as pd

from config import MODEL_DIR

from src.feature_engineering import FeatureEngineer
from src.feature_selection import FeatureSelector


class PredictionEngine:
    """
    Backend prediction engine used by both
    predict.py and Streamlit app.
    """

    def __init__(self):

        self.engineer = FeatureEngineer()

        self.selector = FeatureSelector()

        self.selector.load_features()

        self.engineer.load_scaler()

        self.model = joblib.load(
            MODEL_DIR / "xgboost.pkl"
        )

    ############################################################
    # Predict
    ############################################################

    def predict(self, df):

        ########################################################
        # Save engine ids
        ########################################################

        engine_ids = df["engine_id"].copy()

        ########################################################
        # Feature Engineering
        ########################################################

        df = self.engineer.engineer(df)

        ########################################################
        # Prepare Features
        ########################################################

        X = df.drop(
            columns=[
                "engine_id",
                "cycle",
            ],
            errors="ignore",
        )

        ########################################################
        # Feature Selection
        ########################################################

        X = self.selector.transform(X)

        ########################################################
        # Scaling
        ########################################################

        X = pd.DataFrame(

            self.engineer.transform(X),

            columns=X.columns,

            index=X.index,

        )

        ########################################################
        # Prediction
        ########################################################

        predictions = self.model.predict(X)

        prediction_df = pd.DataFrame({

            "engine_id": engine_ids,

            "Predicted_RUL": predictions

        })

        ########################################################
        # Keep last cycle only
        ########################################################

        prediction_df = (

            prediction_df

            .groupby("engine_id")

            .last()

            .reset_index()

        )

        ########################################################
        # Current cycle
        ########################################################

        current_cycle = (

            df.groupby("engine_id")["cycle"]

            .max()

            .reset_index()

        )

        prediction_df = prediction_df.merge(

            current_cycle,

            on="engine_id",

        )

        ########################################################
        # Estimated failure cycle
        ########################################################

        prediction_df["Failure_Cycle"] = (

            prediction_df["cycle"]

            +

            prediction_df["Predicted_RUL"]

        ).round(0)

        ########################################################
        # Health Status
        ########################################################

        def health(rul):

            if rul >= 80:
                return "🟢 Healthy"

            elif rul >= 40:
                return "🟡 Warning"

            elif rul >= 20:
                return "🟠 Critical"

            else:
                return "🔴 Immediate Maintenance"

        prediction_df["Health"] = (

            prediction_df["Predicted_RUL"]

            .apply(health)

        )

        ########################################################
        # Confidence
        ########################################################

        prediction_df["Confidence"] = (

            prediction_df["Predicted_RUL"]

            .apply(

                lambda x:

                max(

                    55,

                    min(

                        99,

                        round(100 - abs(x) / 3)

                    )

                )

            )

        )

        prediction_df = prediction_df.rename(

            columns={

                "cycle": "Current_Cycle"

            }

        )

        return {
        "predictions": prediction_df,
        "processed_features": X,
}