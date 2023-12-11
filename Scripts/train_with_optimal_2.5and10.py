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
import numpy as np


# Load the training and testing datasets for PM2.5 and PM10
X_train_full = pd.read_csv("Beijing_Air_Pollution_xTrain.csv")
y_train_25_full = pd.read_csv("Beijing_Air_Pollution_2.5yTrain.csv")['PM2.5']
y_train_10_full = pd.read_csv("Beijing_Air_Pollution_10yTrain.csv")['PM10']

X_test_full = pd.read_csv("Beijing_Air_Pollution_xTest.csv")
y_test_25_full = pd.read_csv("Beijing_Air_Pollution_2.5yTest.csv")['PM2.5']
y_test_10_full = pd.read_csv("Beijing_Air_Pollution_10yTest.csv")['PM10']

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test_full)

# Define the models with the best parameters for PM2.5 and PM10

models_best_params_pm25 = {
    'RandomForest': RandomForestRegressor(max_depth=20, n_estimators=100),
    'DecisionTree': DecisionTreeRegressor(max_depth=10),
    'GradientBoosting': GradientBoostingRegressor(learning_rate=0.1, n_estimators=200),
    'NeuralNetwork': MLPRegressor(hidden_layer_sizes=(50,), max_iter=500),
    'PolynomialRegression': Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())]),
    'BayesianRidge': BayesianRidge(alpha_1=1e-06, alpha_2=1e-05),
    'Ridge': Ridge(alpha=10),
    'SGD': SGDRegressor(max_iter=1000, penalty='elasticnet', tol=0.001),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.2),
    'Lasso': Lasso(alpha=0.1)
}

models_best_params_pm10 = {
    'RandomForest': RandomForestRegressor(max_depth=None, n_estimators=100),
    'DecisionTree': DecisionTreeRegressor(max_depth=10),
    'GradientBoosting': GradientBoostingRegressor(learning_rate=0.1, n_estimators=200),
    'Ridge': Ridge(alpha=10),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.2),
    'BayesianRidge': BayesianRidge(alpha_1=1e-06, alpha_2=1e-05),
    'PolynomialRegression': Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())]),
    'SGD': SGDRegressor(max_iter=1000, penalty='l1', tol=0.001),
    'NeuralNetwork': MLPRegressor(hidden_layer_sizes=(50,), max_iter=500)
}

# Function to train and evaluate models
def train_evaluate_models(models_best_params, y_train, y_test):
    results = []
    for model_name, model in models_best_params.items():
        print(f"Training {model_name}...")
        model.fit(X_train_scaled, y_train)

        # Evaluate on training data
        y_train_pred = model.predict(X_train_scaled)
        train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        # Evaluate on testing data
        y_test_pred = model.predict(X_test_scaled)
        test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        results.append({
            'Model': model_name,
            'Train RMSE': train_rmse,
            'Train MAE': train_mae,
            'Train R2': train_r2,
            'Test RMSE': test_rmse,
            'Test MAE': test_mae,
            'Test R2': test_r2
        })
    return results

# Evaluate models for PM2.5
results_pm25 = train_evaluate_models(models_best_params_pm25, y_train_25_full, y_test_25_full)

# Evaluate models for PM10
results_pm10 = train_evaluate_models(models_best_params_pm10, y_train_10_full, y_test_10_full)

# Convert results to DataFrames and display
results_pm25_df = pd.DataFrame(results_pm25)
results_pm10_df = pd.DataFrame(results_pm10)

print("Results for PM2.5:")
print(tabulate(results_pm25_df, headers='keys', tablefmt='pretty', showindex=False))
print("\nResults for PM10:")
print(tabulate(results_pm10_df, headers='keys', tablefmt='pretty', showindex=False))
