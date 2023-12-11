import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso, SGDRegressor, BayesianRidge, ElasticNet, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tabulate import tabulate
from tqdm import tqdm
import numpy as np

# Load the datasets
X_train_full = pd.read_csv("Beijing_Air_Pollution_xTrain.csv")
y_train_25_full = pd.read_csv("Beijing_Air_Pollution_2.5yTrain.csv")['PM2.5']
y_train_10_full = pd.read_csv("Beijing_Air_Pollution_10yTrain.csv")['PM10']

# Select a random 10% subset of the data
subset_fraction = 0.1
subset_size = int(len(X_train_full) * subset_fraction)
X_train_subset = X_train_full.sample(n=subset_size, random_state=42)
y_train_25_subset = y_train_25_full.loc[X_train_subset.index]
y_train_10_subset = y_train_10_full.loc[X_train_subset.index]

# Define the models and their hyperparameters grid
models = {
    'RandomForest': RandomForestRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),
    'DecisionTree': DecisionTreeRegressor(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'BayesianRidge': BayesianRidge(),
    'PolynomialRegression': Pipeline([('poly', PolynomialFeatures()), ('linear', LinearRegression())]),
    'SGD': SGDRegressor(),
    'NeuralNetwork': MLPRegressor()
}

# Expanded hyperparameters for each model
hyperparameters = {
    'RandomForest': {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 4]
    },
    'GradientBoosting': {
        'n_estimators': [100, 200, 300, 400],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9]
    },
    'DecisionTree': {
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 4, 6, 8],
        'min_samples_leaf': [1, 2, 4]
    },
    'Ridge': {
        'alpha': [0.1, 1, 10, 100]
    },
    'Lasso': {
        'alpha': [0.1, 1, 10, 100]
    },
    'ElasticNet': {
        'alpha': [0.1, 1, 10, 100],
        'l1_ratio': [0.2, 0.5, 0.7, 0.9]
    },
    'BayesianRidge': {
        'alpha_1': [1e-6, 1e-5, 1e-4],
        'alpha_2': [1e-6, 1e-5, 1e-4]
    },
    'PolynomialRegression': {
        'model': Pipeline([('poly', PolynomialFeatures()), ('linear', LinearRegression())]),
        'params': {'poly__degree': [2, 3]}
    },
    'SGD': {
        'penalty': ['l2', 'l1', 'elasticnet'],
        'max_iter': [1000, 2000],
        'tol': [1e-3, 1e-4]
    },
    'NeuralNetwork': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
        'max_iter': [500, 1000]
    }
}

# Cross-validation configuration
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

def grid_search_model(model, params, target):
    # Using half of the cores available on an i9-13900K
    n_jobs_cores = 12  # Adjust this based on your observations
    gs = GridSearchCV(model, params, cv=kfold, scoring='neg_mean_squared_error', n_jobs=n_jobs_cores, verbose=3)
    gs.fit(X_train_subset, target)
    return gs.best_estimator_, gs.best_params_

# Function to perform model selection and evaluation
def evaluate_models(models, hyperparameters, target):
    results = []
    for model_name in tqdm(models.keys(), desc=f"Processing models for {target.name}"):
        model = models[model_name]
        params = hyperparameters[model_name]
        print(f"Processing {model_name} for {target.name}...")
        best_model, best_params = grid_search_model(model, params, target)
        y_pred = best_model.predict(X_train_subset)
        rmse = mean_squared_error(target, y_pred, squared=False)
        mae = mean_absolute_error(target, y_pred)
        r2 = r2_score(target, y_pred)
        results.append({
            'Model': model_name,
            'Best Params': best_params,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        })
    return results

# Evaluate models for PM2.5
results_pm25_subset = evaluate_models(models, hyperparameters, y_train_25_subset)

# Evaluate models for PM10
results_pm10_subset = evaluate_models(models, hyperparameters, y_train_10_subset)

# Convert results to DataFrames
results_pm25_subset_df = pd.DataFrame(results_pm25_subset)
results_pm10_subset_df = pd.DataFrame(results_pm10_subset)

# Display results
print("Results for PM2.5 on Subset:")
print(tabulate(results_pm25_subset_df, headers='keys', tablefmt='pretty', showindex=False))
print("\nResults for PM10 on Subset:")
print(tabulate(results_pm10_subset_df, headers='keys', tablefmt='pretty', showindex=False))

# Export the results to CSV files
results_pm25_subset_df.to_csv("C:/Users/Tianyi Zhang/Desktop/results_pm25.csv", index=False)
results_pm10_subset_df.to_csv("C:/Users/Tianyi Zhang/Desktop/results_pm10.csv", index=False)