# House Price Prediction Using Machine Learning

This VS Code project follows the uploaded minor-project report.

## Project features

- Generates 1,000 synthetic property records using NumPy and Pandas.
- Uses five input features:
  - `Area_sqft`
  - `Bedrooms`
  - `Bathrooms`
  - `Property_Age`
  - `Location_Score`
- Predicts the continuous target variable `Price`.
- Displays dataset preview, shape, information, summary statistics and missing values.
- Creates EDA graphs: price distribution, area distribution, area vs price, and correlation heatmap.
- Trains and compares:
  1. Linear Regression
  2. Decision Tree Regressor
  3. Random Forest Regressor
- Evaluates each model using MAE, MSE, RMSE and R-squared.
- Selects the model with the lowest test-set RMSE.
- Accepts new house details and predicts an illustrative price.
- Saves the dataset, model comparison, graphs and trained model.

## Requirements

- Python 3.10 or later recommended
- Visual Studio Code
- Python extension for VS Code

## Run in VS Code (Windows)

1. Extract the ZIP file.
2. Open the extracted `house_price_prediction_project` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create a virtual environment:

   ```powershell
   py -m venv .venv
   ```

5. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt:

   ```cmd
   .venv\Scripts\activate.bat
   ```

6. Install the libraries:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

7. Run the project:

   ```powershell
   python main.py
   ```

8. When prompted, enter values within the displayed ranges. Example:
   - Area: `1500`
   - Bedrooms: `3`
   - Bathrooms: `2`
   - Property age: `5`
   - Location score: `8`

## Output files

After the program runs, it creates:

- `data/house_price_dataset.csv`
- `outputs/model_performance_comparison.csv`
- `outputs/plots/price_distribution.png`
- `outputs/plots/area_distribution.png`
- `outputs/plots/area_vs_price.png`
- `outputs/plots/correlation_heatmap.png`
- `outputs/plots/model_comparison_rmse.png`
- `models/best_house_price_model.joblib`

The actual metric values and prediction are produced by running the program. Do not copy assumed values into the report.

## Important academic note

The report describes a synthetic dataset generated for demonstration. The prices are produced by an illustrative formula with random noise; they are **not real property transaction prices** and do not represent the market value of a house in any city. The output must not be presented as a professional valuation.

## Report alignment

The feature names, sample size, synthetic-data formula, three algorithms, 80:20 train-test split, evaluation metrics and lowest-RMSE model-selection rule follow the submitted report. This version adds input validation, organized output folders and saving the trained model to make it easier to run in VS Code.
