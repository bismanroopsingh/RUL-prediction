import os

import pandas as pd
import matplotlib.pyplot as plt

from config import OUTPUT_DIR


def main():

    print("=" * 60)
    print("Prediction Error Analysis")
    print("=" * 60)

    ############################################################
    # Load Prediction Results
    ############################################################

    results = pd.read_csv(
        OUTPUT_DIR / "prediction_results.csv"
    )

    ############################################################
    # Calculate Errors
    ############################################################

    results["Residual"] = (

        results["Actual RUL"]

        - results["Predicted RUL"]

    )

    results["Absolute Error"] = (

        results["Residual"]

        .abs()

    )

    ############################################################
    # Save Error CSV
    ############################################################

    results.to_csv(

        OUTPUT_DIR / "prediction_errors.csv",

        index=False,

    )

    ############################################################
    # Top 20 Largest Errors
    ############################################################

    largest_errors = (

        results

        .sort_values(

            "Absolute Error",

            ascending=False,

        )

        .head(20)

    )

    largest_errors.to_csv(

        OUTPUT_DIR / "largest_errors.csv",

        index=False,

    )

    ############################################################
    # Actual vs Predicted
    ############################################################

    plt.figure(figsize=(8,8))

    plt.scatter(

        results["Actual RUL"],

        results["Predicted RUL"],

        alpha=0.7,

    )

    max_value = max(

        results["Actual RUL"].max(),

        results["Predicted RUL"].max(),

    )

    plt.plot(

        [0, max_value],

        [0, max_value],

        linestyle="--",

    )

    plt.xlabel("Actual RUL")

    plt.ylabel("Predicted RUL")

    plt.title("Actual vs Predicted RUL")

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / "actual_vs_predicted.png",

        dpi=300,

    )

    plt.close()

    ############################################################
    # Residual Plot
    ############################################################

    plt.figure(figsize=(8,6))

    plt.scatter(

        results["Predicted RUL"],

        results["Residual"],

        alpha=0.7,

    )

    plt.axhline(

        0,

        linestyle="--",

    )

    plt.xlabel("Predicted RUL")

    plt.ylabel("Residual")

    plt.title("Residual Plot")

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / "residual_plot.png",

        dpi=300,

    )

    plt.close()

    ############################################################
    # Residual Histogram
    ############################################################

    plt.figure(figsize=(8,6))

    plt.hist(

        results["Residual"],

        bins=30,

    )

    plt.xlabel("Residual")

    plt.ylabel("Frequency")

    plt.title("Residual Distribution")

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / "residual_histogram.png",

        dpi=300,

    )

    plt.close()

    ############################################################
    # Absolute Error Histogram
    ############################################################

    plt.figure(figsize=(8,6))

    plt.hist(

        results["Absolute Error"],

        bins=30,

    )

    plt.xlabel("Absolute Error")

    plt.ylabel("Frequency")

    plt.title("Absolute Error Distribution")

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / "absolute_error_histogram.png",

        dpi=300,

    )

    plt.close()

    ############################################################
    # Error Statistics
    ############################################################

    summary = pd.DataFrame({

        "Metric":[

            "Mean Error",

            "Median Error",

            "Std Error",

            "Maximum Error",

            "Minimum Error",

            "Mean Absolute Error",

            "Median Absolute Error",

        ],

        "Value":[

            results["Residual"].mean(),

            results["Residual"].median(),

            results["Residual"].std(),

            results["Residual"].max(),

            results["Residual"].min(),

            results["Absolute Error"].mean(),

            results["Absolute Error"].median(),

        ]

    })

    summary.to_csv(

        OUTPUT_DIR / "error_statistics.csv",

        index=False,

    )

    ############################################################
    # Console Output
    ############################################################

    print()

    print("=" * 60)
    print("Largest Prediction Errors")
    print("=" * 60)

    print(

        largest_errors[

            [

                "Engine",

                "Actual RUL",

                "Predicted RUL",

                "Absolute Error",

            ]

        ]

    )

    print()

    print("=" * 60)
    print("Files Generated")
    print("=" * 60)

    print("✔ prediction_errors.csv")
    print("✔ largest_errors.csv")
    print("✔ error_statistics.csv")
    print("✔ actual_vs_predicted.png")
    print("✔ residual_plot.png")
    print("✔ residual_histogram.png")
    print("✔ absolute_error_histogram.png")


if __name__ == "__main__":
    main()