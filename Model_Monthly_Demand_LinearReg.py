import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error

# Load and prepare data
df = pd.read_csv("C:/_VAMK/_Coding/Data/Input/Combined Monthly Data 2020-2024.csv")
dates = pd.date_range(start='2020-01-01', periods=len(df), freq='ME')
df['Date'] = dates
df.set_index('Date', inplace=True)

# Define features and target
X = df[['Avg_temperature','Max_temperature']]#,'Min_temperature','Sun_Duration'
y = df['Demand_GWh']

# Train-test split
train_size = int(len(df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
dates_train, dates_test = df.index[:train_size], df.index[train_size:]

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Residuals
residuals = y_test - y_test_pred
standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)

# Evaluation metrics
print("Linear Regression Model Evaluation:")
print(f" - MSE:  {mean_squared_error(y_test, y_test_pred):.3f}")
print(f" - RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.3f}")
print(f" - MAE:  {mean_absolute_error(y_test, y_test_pred):.3f}")
print(f" - MAPE: {mean_absolute_percentage_error(y_test, y_test_pred):.3f}")
print(f" - R²:   {r2_score(y_test, y_test_pred):.3f}")

# Coefficients
print("\nModel Coefficients:")
for name, coef in zip(X.columns, model.coef_):
    print(f" - {name}: {coef:.4f}")
print(f" - Intercept: {model.intercept_:.4f}")

residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

# Combine for full residual analysis
residuals_all = pd.concat([residuals_train, residuals_test])
predictions_all = pd.concat([pd.Series(y_train_pred, index=dates_train),
                             pd.Series(y_test_pred, index=dates_test)])
standardized_residuals_all = (residuals_all - residuals_all.mean()) / residuals_all.std()
dates_all = df.index  # full date index

# === PLOTS === #

# 1. Actual vs Predicted (Test)
plt.figure(figsize=(5, 5))
plt.scatter(y, predictions_all, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Demand (GWh)")
plt.ylabel("Predicted Demand (GWh)")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Combined Forecast Plot
plt.figure(figsize=(12, 6))
plt.plot(dates_train, y_train, label='Training Data (Actual)', color='blue')
plt.plot(dates_train, y_train_pred, label='Fitted Values (Train)', color='green', linestyle='--')
plt.plot(dates_test, y_test, label='Test Data (Actual)', color='orange')
plt.plot(dates_test, y_test_pred, label='Forecast (Test)', color='red', linestyle='--')
plt.title("Monthly Electricity Demand Forecast\n(Linear Regression Model)")
plt.xlabel("Date")
plt.ylabel("Demand (GWh)")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# === RESIDUAL ANALYSIS === #
# 3. Standardized Residuals vs Fitted
plt.figure(figsize=(8, 5))
plt.scatter(predictions_all, standardized_residuals_all, edgecolors='k')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values (Predicted)")
plt.ylabel("Standardized Residuals")
plt.title("Standardized Residuals vs Fitted Values")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Standardized Residuals vs Date
plt.figure(figsize=(10, 5))
plt.plot(dates_all, standardized_residuals_all, marker='o', linestyle='-', color='purple')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Date")
plt.ylabel("Standardized Residuals")
plt.title("Standardized Residuals vs Date")
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Histogram of Residuals
plt.figure(figsize=(5, 5))
plt.hist(residuals_all, bins=10, edgecolor='black', alpha=0.7)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Q-Q Plot of Residuals
plt.figure(figsize=(5, 5))
stats.probplot(residuals_all, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.grid(True)
plt.tight_layout()
plt.show()
