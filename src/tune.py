import json
import os

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

from optuna.integration import XGBoostPruningCallback
class XGBoostTuner:
    """
    Hyperparameter tuning for XGBoost using Optuna
    and GroupKFold cross-validation.
    """

    def __init__(
        self,
        X,
        y,
        groups,
        n_splits=5,
        random_state=42,
    ):

        self.X = X
        self.y = y
        self.groups = groups

        self.n_splits = n_splits
        self.random_state = random_state

        self.study = None
        self.best_params = None

    ############################################################
    # Objective Function
    ############################################################

    def objective(self, trial):

        params = {

            "objective": "reg:squarederror",

            "n_estimators": trial.suggest_int(
                "n_estimators",
                200,
                800,
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.15,
                log=True,
            ),

            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                12,
            ),

            "subsample": trial.suggest_float(
                "subsample",
                0.6,
                1.0,
            ),

            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0,
            ),

            "gamma": trial.suggest_float(
                "gamma",
                0.0,
                5.0,
            ),

            "min_child_weight": trial.suggest_int(
                "min_child_weight",
                1,
                10,
            ),

            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                0.0,
                10.0,
            ),

            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                0.0,
                10.0,
            ),

            "max_delta_step": trial.suggest_int(
                "max_delta_step",
                0,
                10,
            ),

            "random_state": self.random_state,

            "n_jobs": -1,
        }

        gkf = GroupKFold(
            n_splits=self.n_splits
        )

        rmse_scores = []

        for train_idx, valid_idx in gkf.split(
            self.X,
            self.y,
            self.groups,
        ):

            X_train = self.X.iloc[train_idx]
            X_valid = self.X.iloc[valid_idx]

            y_train = self.y.iloc[train_idx]
            y_valid = self.y.iloc[valid_idx]

            model = xgb.XGBRegressor(**params)

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False,
            )
            predictions = model.predict(
                X_valid
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_valid,
                    predictions,
                )
            )

            rmse_scores.append(rmse)

        return np.mean(rmse_scores)

    ############################################################
    # Run Optimization
    ############################################################

    def optimize(
        self,
        n_trials=50,
    ):

        print("=" * 60)
        print("Starting Optuna Optimization...")
        print("=" * 60)

        sampler = optuna.samplers.TPESampler(
            seed=self.random_state
        )

        self.study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name="NASA_RUL_XGBoost",
        )

        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            show_progress_bar=True,
        )

        self.best_params = self.study.best_params

        print("\nOptimization Finished")

        print(
            f"Best RMSE : {self.study.best_value:.4f}"
        )
        print("\nBest Parameters:")

        for key, value in self.best_params.items():
            print(f"{key}: {value}")
        self.save_best_params()
        self.save_history()
        return self.study

    ############################################################
    # Save Best Parameters
    ############################################################

    def save_best_params(
        self,
        path="models/best_params.json",
    ):

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True,
        )

        with open(path, "w") as f:

            json.dump(
                self.best_params,
                f,
                indent=4,
            )

        print(
            f"Best parameters saved to {path}"
        )

    ############################################################
    # Save Optimization History
    ############################################################

    def save_history(
        self,
        path="outputs/optuna_history.csv",
    ):

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True,
        )

        history = self.study.trials_dataframe()

        history.to_csv(
            path,
            index=False,
        )

        print(
            f"Optimization history saved to {path}"
        )

    ############################################################
    # Get Best Model
    ############################################################

    def get_best_model(self):

        params = self.best_params.copy()

        params.update({

            "objective": "reg:squarederror",

            "random_state": self.random_state,

            "n_jobs": -1,

        })

        model = xgb.XGBRegressor(
            **params
        )

        model.fit(
            self.X,
            self.y,
        )
        importance = pd.DataFrame({
            "Feature": self.X.columns,
            "Importance": model.feature_importances_
        })

        importance = importance.sort_values(
            "Importance",
            ascending=False
        )

        os.makedirs("outputs", exist_ok=True)

        importance.to_csv(
            "outputs/feature_importance.csv",
            index=False,
        )
        return model