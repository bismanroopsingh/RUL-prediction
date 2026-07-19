import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.metrics import explained_variance_score

class Evaluator:
    """
    Evaluation module for NASA C-MAPSS RUL prediction.
    """

    def __init__(self, output_dir="outputs"):

        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

    ############################################################
    # NASA Score
    ############################################################

    @staticmethod
    def nasa_score(y_true, y_pred):

        score = 0

        for actual, pred in zip(y_true, y_pred):

            d = pred - actual

            if d < 0:

                score += np.exp(-d / 13) - 1

            else:

                score += np.exp(d / 10) - 1

        return score

    ############################################################
    # Metrics
    ############################################################

    def calculate_metrics(self, y_true, y_pred):

        metrics = {

            "MAE":
            mean_absolute_error(
                y_true,
                y_pred,
            ),

            "RMSE":
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            ),
            "Explained Variance":
            explained_variance_score(
                y_true,
                y_pred,
            ),
            "Max Error":
                max_error(
                    y_true,
                    y_pred,
                ),



            "R2":
            r2_score(
                y_true,
                y_pred,
            ),

            "NASA Score":
            self.nasa_score(
                y_true,
                y_pred,
            ),
        }

        return metrics

    ############################################################
    # Save Metrics
    ############################################################

    def save_metrics(self, metrics):

        df = pd.DataFrame([metrics])

        df.to_csv(
            os.path.join(
                self.output_dir,
                "metrics.csv",
            ),
            index=False,
        )

        with open(
            os.path.join(
                self.output_dir,
                "metrics.json",
            ),
            "w",
        ) as f:

            metrics = {
                k: float(v)
                for k, v in metrics.items()
            }

            json.dump(
                metrics,
                f,
                indent=4,
            )

    ############################################################
    # Save Predictions
    ############################################################

    def save_predictions(
        self,
        y_true,
        y_pred,
    ):

        pred = pd.DataFrame({

            "Actual_RUL": y_true,

            "Predicted_RUL": y_pred,

            "Residual":
            y_true - y_pred,

        })
        pred["Absolute_Error"] = (
            pred["Actual_RUL"] -
            pred["Predicted_RUL"]
        ).abs()

        pred = pred.sort_values(
            "Absolute_Error",
            ascending=False
        )


        pred.to_csv(

            os.path.join(

                self.output_dir,

                "test_predictions.csv",

            ),

            index=False,

        )

    ############################################################
    # Actual vs Predicted
    ############################################################

    def plot_predictions(
        self,
        y_true,
        y_pred,
    ):

        plt.figure(figsize=(8,6))

        plt.scatter(
            y_true,
            y_pred,
            alpha=0.7,
        )
        plt.xlim(0, max(y_true) + 5)
        plt.ylim(0, max(y_true) + 5)
        plt.plot(

            [0, max(y_true)],

            [0, max(y_true)],

            "r--",

        )

        plt.xlabel("Actual RUL")

        plt.ylabel("Predicted RUL")

        plt.title("Actual vs Predicted")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                self.output_dir,

                "actual_vs_predicted.png",

            )

        )

        plt.close()

    ############################################################
    # Residual Plot
    ############################################################

    def plot_residuals(
        self,
        y_true,
        y_pred,
    ):

        residual = y_true - y_pred

        plt.figure(figsize=(8,6))

        plt.scatter(
            y_pred,
            residual,
            alpha=0.5,
        )

        plt.axhline(
            0,
            color="red",
            linestyle="--",
        )

        plt.grid(True)

        plt.xlabel("Predicted RUL")

        plt.ylabel("Residual")

        plt.title("Residual Plot")

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                self.output_dir,

                "residual_plot.png",

            )

        )

        plt.close()

    ############################################################
    # Residual Histogram
    ############################################################

    def plot_error_distribution(
        self,
        y_true,
        y_pred,
    ):

        residual = y_true - y_pred

        plt.figure(figsize=(8,6))

        plt.hist(
            residual,
            bins=30,
        )

        plt.xlabel("Residual")

        plt.ylabel("Count")

        plt.title("Residual Distribution")

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                self.output_dir,

                "error_distribution.png",

            )

        )

        plt.close()
        
    def plot_feature_importance(
        self,
        csv_path="outputs/feature_importance.csv",
    ):

        if not os.path.exists(csv_path):
            return

        importance = pd.read_csv(csv_path)

        importance = importance.head(20)

        plt.figure(figsize=(10,8))

        plt.barh(
            importance["Feature"],
            importance["Importance"]
        )

        plt.gca().invert_yaxis()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_dir,
                "feature_importance.png",
            )
        )

        plt.close()

    ############################################################
    # Complete Evaluation
    ############################################################

    def evaluate(
        self,
        y_true,
        y_pred,
    ):

        metrics = self.calculate_metrics(
            y_true,
            y_pred,
        )

        self.save_metrics(metrics)

        self.save_predictions(
            y_true,
            y_pred,
        )

        self.plot_predictions(
            y_true,
            y_pred,
        )

        self.plot_residuals(
            y_true,
            y_pred,
        )

        self.plot_error_distribution(
            y_true,
            y_pred,
        )
        self.plot_feature_importance()
        print("=" * 60)

        print("Evaluation Results")

        print("=" * 60)

        for k, v in metrics.items():

            print(f"{k:<15}: {v:.4f}")

        return metrics