import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
import matplotlib.ticker as mtick

def get_season(month):
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'autumn'

# Load and prepare data
df = pd.read_csv("C:/_VAMK/_Coding/Data/Input/Daily Combined data Vaasa 2020-2024.csv")
dates = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
df['Date'] = dates
df['Is_workday'] = df['Iso_Weekday'].apply(lambda x: 1 if x <= 5 else 0)
df.set_index('Date', inplace=True)
df['Season'] = df.index.month.map(get_season)

# One-hot encode Month_No
df_encoded = pd.get_dummies(df, columns=['Month_No'], drop_first=True)

# Define feature set
base_features = ['Avg_temperature', 'Min_temperature', 'Max_temperature', 'Sun_Duration', 'Is_workday']
month_cols = [col for col in df_encoded.columns if col.startswith('Month_No_')]
X = df_encoded[base_features + month_cols]
y = df_encoded['Demand_MWh']

# Train/test split
train_size = int(len(df_encoded) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
dates_train, dates_test = df_encoded.index[:train_size], df_encoded.index[train_size:]

# Train XGBoost model
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
xgb.fit(X_train, y_train)

# Predict
y_train_pred = xgb.predict(X_train)
y_test_pred = xgb.predict(X_test)

# Evaluation
print("XGBoost Model Evaluation (Test Set):")
print(f" - MSE:  {mean_squared_error(y_test, y_test_pred):.2f}")
print(f" - RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.2f}")
print(f" - MAE:  {mean_absolute_error(y_test, y_test_pred):.2f}")
print(f" - MAPE: {mean_absolute_percentage_error(y_test, y_test_pred):.4f}")
print(f" - R²:   {r2_score(y_test, y_test_pred):.3f}")

# Residuals
predictions_all = pd.Series(np.concatenate([y_train_pred, y_test_pred]), index=df_encoded.index)
residuals_all = y - predictions_all
standardized_residuals_all = (residuals_all - residuals_all.mean()) / residuals_all.std()

# === PLOTS === #

# 1. Actual vs Predicted
plt.figure(figsize=(6, 5))
plt.scatter(y, predictions_all, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Demand (MWh)")
plt.ylabel("Predicted Demand (MWh)")
plt.title("XGBoost: Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Forecast Plot (Test)
plt.plot(dates_test, y_test, label='Test Data (Actual)', color='orange')
plt.plot(dates_test, y_test_pred, label='Forecast (Test)', color='red', linestyle='--')
plt.title("Daily Electricity Demand Forecast - XGBoost")
plt.xlabel("Date")
plt.ylabel("Demand (MWh)")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Forecast Plot (Train + Test)
plt.figure(figsize=(12, 6))
plt.plot(dates_train, y_train, label='Train Actual', color='blue')
plt.plot(dates_train, y_train_pred, label='Train Predicted', color='cyan', linestyle='--')
plt.plot(dates_test, y_test, label='Test Actual', color='orange')
plt.plot(dates_test, y_test_pred, label='Test Predicted', color='red', linestyle='--')
plt.title("Daily Electricity Demand Forecast - XGBoost")
plt.xlabel("Date")
plt.ylabel("Demand (MWh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === RESIDUAL ANALYSIS === #

# 4. Standardized Residuals vs Fitted
plt.figure(figsize=(8, 5))
plt.scatter(predictions_all, standardized_residuals_all, edgecolors='k')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values (Predicted)")
plt.ylabel("Standardized Residuals")
plt.title("Standardized Residuals vs Fitted Values")
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Residuals vs Date
plt.figure(figsize=(10, 4))
plt.plot(df_encoded.index, standardized_residuals_all, color='purple', marker='o', linestyle='-')
plt.axhline(0, color='red', linestyle='--')
plt.title("Standardized Residuals vs Date")
plt.xlabel("Date")
plt.ylabel("Standardized Residuals")
plt.tight_layout()
plt.grid(True)
plt.show()

# 6. Histogram of Residuals
plt.figure(figsize=(6, 4))
sns.histplot(residuals_all, bins=20, kde=True, edgecolor='black')
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.tight_layout()
plt.show()

# 7. Q-Q Plot
plt.figure(figsize=(5, 5))
stats.probplot(residuals_all, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.grid(True)
plt.tight_layout()
plt.show()

# === FEATURE IMPORTANCE TABLE ===
importance_scores = xgb.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance_scores
}).sort_values(by='Importance', ascending=False)

# Print as a table
print("\nFeature Importance Table (XGBoost):")
print(importance_df.to_string(index=False))

# === Feature Importance ===
plt.figure(figsize=(10, 6))
plot_importance(xgb, importance_type='gain', height=0.4, grid=True, show_values=False)
plt.title("Feature Importances from XGBoost Regressor")
plt.tight_layout()
plt.show()
