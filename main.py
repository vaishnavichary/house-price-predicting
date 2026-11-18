"""
HOUSE PRICE PREDICTION USING MACHINE LEARNING
Based on the submitted project report.

Dataset: 1,000 synthetic/illustrative records.
Features: Area_sqft, Bedrooms, Bathrooms, Property_Age, Location_Score
Target: Price
Models: Linear Regression, Decision Tree Regressor, Random Forest Regressor
"""

from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
MODEL_DIR = ROOT / "models"

FEATURES = [
    "Area_sqft",
    "Bedrooms",
    "Bathrooms",
    "Property_Age",
    "Location_Score",
]
TARGET = "Price"


def create_directories():
    """Create folders used to save generated files."""
    for folder in (DATA_DIR, OUTPUT_DIR, PLOT_DIR, MODEL_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def generate_dataset(n=1000, random_state=42):
    """
    Generate the illustrative synthetic dataset described in the report.
    These generated prices are not actual market transaction prices.
    """
    rng = np.random.RandomState(random_state)

    area = rng.randint(500, 3500, n)
    bedrooms = rng.randint(1, 6, n)
    bathrooms = rng.randint(1, 5, n)
    age = rng.randint(0, 31, n)
    location_score = rng.randint(1, 11, n)

    noise = rng.normal(0, 800000, n)
    price = (
        2500000
        + area * 4500
        + bedrooms * 350000
        + bathrooms * 500000
        - age * 45000
        + location_score * 600000
        + noise
    )
    price = np.maximum(price, 500000)

    return pd.DataFrame({
        "Area_sqft": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Property_Age": age,
        "Location_Score": location_score,
        "Price": price,
    })


def inspect_dataset(data):
    """Print basic dataset details and missing-value counts."""
    print("\n========== DATASET PREVIEW ==========")
    print(data.head())
    print("\n========== DATASET SHAPE ==========")
    print(data.shape)
    print("\n========== DATASET INFORMATION ==========")
    data.info()
    print("\n========== STATISTICAL SUMMARY ==========")
    print(data.describe())
    print("\n========== MISSING VALUES ==========")
    print(data.isnull().sum())


def create_eda_plots(data):
    """Create the plots described in the project report."""
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    plt.hist(data[TARGET], bins=25, edgecolor="black")
    plt.title("House Price Distribution")
    plt.xlabel("Price (illustrative units)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "price_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(data["Area_sqft"], bins=25, edgecolor="black")
    plt.title("Area Distribution")
    plt.xlabel("Area (square feet)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "area_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(data["Area_sqft"], data[TARGET], alpha=0.5)
    plt.title("Area vs House Price")
    plt.xlabel("Area (square feet)")
    plt.ylabel("Price (illustrative units)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "area_vs_price.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 7))
    sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()


def train_and_evaluate(data):
    """Train three regressors and compare them on the same test set."""
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, random_state=42, n_jobs=-1
        ),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        trained_models[name] = model

        mse = mean_squared_error(y_test, predictions)
        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, predictions),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R2": r2_score(y_test, predictions),
        })

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    results_df.to_csv(OUTPUT_DIR / "model_performance_comparison.csv", index=False)

    print("\n========== MODEL COMPARISON ==========")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))

    best_model_name = results_df.loc[0, "Model"]
    best_model = trained_models[best_model_name]
    print(f"\nSelected model (lowest test RMSE): {best_model_name}")

    # Save the selected model and feature order for later use.
    model_bundle = {
        "model": best_model,
        "model_name": best_model_name,
        "features": FEATURES,
        "target": TARGET,
        "disclaimer": "Trained on synthetic illustrative data; not a real-market valuation model.",
    }
    joblib.dump(model_bundle, MODEL_DIR / "best_house_price_model.joblib")

    # Save model comparison chart.
    plt.figure(figsize=(8, 5))
    plt.bar(results_df["Model"], results_df["RMSE"])
    plt.title("Model Comparison - RMSE (lower is better)")
    plt.ylabel("RMSE")
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "model_comparison_rmse.png", dpi=150)
    plt.close()

    return best_model, best_model_name, results_df


def get_validated_house_input():
    """Read and validate a new property's features from the terminal."""
    print("\nEnter the details of a house to estimate its illustrative price.")
    print("Location Score is a numeric score from 1 to 10 used by this sample dataset.")

    while True:
        try:
            area = float(input("Area in square feet (500-3499): "))
            bedrooms = int(input("Number of bedrooms (1-5): "))
            bathrooms = int(input("Number of bathrooms (1-4): "))
            age = int(input("Property age in years (0-30): "))
            location_score = int(input("Location score (1-10): "))

            if not 500 <= area < 3500:
                raise ValueError("Area must be from 500 up to (but not including) 3500.")
            if not 1 <= bedrooms <= 5:
                raise ValueError("Bedrooms must be between 1 and 5.")
            if not 1 <= bathrooms <= 4:
                raise ValueError("Bathrooms must be between 1 and 4.")
            if not 0 <= age <= 30:
                raise ValueError("Property age must be between 0 and 30.")
            if not 1 <= location_score <= 10:
                raise ValueError("Location score must be between 1 and 10.")

            return pd.DataFrame([{
                "Area_sqft": area,
                "Bedrooms": bedrooms,
                "Bathrooms": bathrooms,
                "Property_Age": age,
                "Location_Score": location_score,
            }], columns=FEATURES)

        except ValueError as error:
            print(f"Invalid input: {error}")
            print("Please enter the values again.\n")


def main():
    create_directories()

    print("=" * 62)
    print("       HOUSE PRICE PREDICTION USING MACHINE LEARNING")
    print("=" * 62)
    print("Creating 1,000 synthetic illustrative property records...")

    data = generate_dataset()
    dataset_path = DATA_DIR / "house_price_dataset.csv"
    data.to_csv(dataset_path, index=False)
    print(f"Dataset saved to: {dataset_path.relative_to(ROOT)}")

    inspect_dataset(data)
    create_eda_plots(data)
    print(f"\nEDA graphs saved in: {PLOT_DIR.relative_to(ROOT)}")

    best_model, best_model_name, results_df = train_and_evaluate(data)

    new_house = get_validated_house_input()
    predicted_price = best_model.predict(new_house)[0]

    print("\n========== PREDICTION RESULT ==========")
    print(f"Selected model : {best_model_name}")
    print(f"Estimated price: {predicted_price:,.2f} illustrative units")
    print("\nNote: This is an academic demonstration using synthetic data.")
    print("It is not an actual market price or professional property valuation.")

    print("\nGenerated files:")
    print("- data/house_price_dataset.csv")
    print("- outputs/model_performance_comparison.csv")
    print("- outputs/plots/*.png")
    print("- models/best_house_price_model.joblib")


if __name__ == "__main__":
    main()
