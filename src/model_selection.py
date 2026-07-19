import os
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

import xgboost as xgb


class ModelBenchmark:
    """
    Benchmark multiple regression models for
    Remaining Useful Life prediction.
    """

    def __init__(self):

        self.models = {

            "Random Forest": RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
            ),

            "Extra Trees": ExtraTreesRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
            ),

            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=300,
                random_state=42,
            ),

            "XGBoost": xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            ),
        }

        self.best_model = None
        self.best_model_name = None
        self.results = None

    #############################################################
    # Benchmark Models
    #############################################################
    def evaluate(
        self,
        X_train,
        X_valid,
        y_train,
        y_valid,
    ):

        results = []

        best_rmse = float("inf")

        for name, model in self.models.items():

            print("=" * 60)
            print(f"Training {name}")

            start_train = time.time()

            model.fit(
                X_train,
                y_train,
            )

            train_time = (
                time.time()
                - start_train
            )

            start_pred = time.time()

            pred = model.predict(
                X_valid
            )

            prediction_time = (
                time.time()
                - start_pred
            )

            mae = mean_absolute_error(
                y_valid,
                pred,
            )

            rmse = (
                mean_squared_error(
                    y_valid,
                    pred,
                )
                ** 0.5
            )

            r2 = r2_score(
                y_valid,
                pred,
            )

            mape = (
                mean_absolute_percentage_error(
                    y_valid,
                    pred,
                )
                * 100
            )

            results.append({

                "Model": name,

                "MAE": round(mae, 3),

                "RMSE": round(rmse, 3),

                "MAPE (%)": round(
                    mape,
                    2,
                ),

                "R2": round(
                    r2,
                    4,
                ),

                "Training Time (s)": round(
                    train_time,
                    2,
                ),

                "Prediction Time (s)": round(
                    prediction_time,
                    4,
                ),
            })

            if rmse < best_rmse:

                best_rmse = rmse

                self.best_model = model

                self.best_model_name = name

        self.results = (
            pd.DataFrame(results)
            .sort_values("RMSE")
            .reset_index(drop=True)
        )

        self.results.insert(
            0,
            "Rank",
            range(
                1,
                len(self.results) + 1,
            ),
        )

        return self.results

    #############################################################
    # Save Results
    #############################################################

    def save_results(
        self,
        output_dir="outputs",
    ):

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        self.results.to_csv(

            os.path.join(
                output_dir,
                "model_comparison.csv",
            ),

            index=False,
        )

        print(
            "Benchmark results saved."
        )

    #############################################################
    # Save Best Model
    #############################################################

    def save_best_model(
        self,
        path="models/best_model.pkl",
    ):

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True,
        )

        joblib.dump(
            self.best_model,
            path,
        )

        print(
            f"Best model saved : {self.best_model_name}"
        )

    #############################################################
    # Plot Comparison
    #############################################################

    def plot_results(
        self,
        output_dir="outputs",
    ):

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        plt.figure(figsize=(8,5))

        plt.bar(
            self.results["Model"],
            self.results["RMSE"],
        )

        plt.ylabel("RMSE")

        plt.title("Model Comparison")

        plt.xticks(rotation=20)

        plt.tight_layout()

        plt.savefig(

            os.path.join(
                output_dir,
                "model_comparison.png",
            )

        )

        plt.close()

        print(
            "Comparison plot saved."
        )