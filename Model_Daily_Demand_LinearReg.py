import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error

'''
def get_season(month):
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:  # 9, 10, 11
        return 'autumn'
'''
    
# Load and prepare data
df = pd.read_csv("C:/_VAMK/_Coding/Data/Input/Daily Combined data Vaasa 2020-2024.csv")
dates = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
df['Date'] = dates
# Create binary workday feature
df['Is_workday'] = df['Iso_Weekday'].apply(lambda x: 1 if x <= 5 else 0)
df.set_index('Date', inplace=True)

#df['Season'] = df.index.month.map(get_season)
#df['season'] = pd.Categorical(df['Season'], categories=['winter', 'spring', 'summer', 'autumn'], ordered=True)

# One-hot encode categorical variables
categorical_vars = ['Month_No']#'Season', 'Iso_Weekday'
df_encoded = pd.get_dummies(df, columns=categorical_vars, drop_first=True)

base_features = ['Avg_temperature','Min_temperature','Max_temperature','Sun_Duration', 'Is_workday']
# Add all encoded Month column
month_cols = [col for col in df_encoded.columns if col.startswith('Month_No_')]
# Add all encoded Weekday column
#weekday_cols = [col for col in df_encoded.columns if col.startswith('Iso_Weekday_')]
# Add all encoded Weekday column
#season_cols = [col for col in df_encoded.columns if col.startswith('Season_')]

X = df_encoded[base_features + month_cols]#season_cols + weekday_cols
y = df_encoded['Demand_MWh']

# Train-test split
train_size = int(len(df)*0.8)
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
print("Multiple Linear Regression Model Evaluation:")
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

# Add intercept column manually for statsmodels
X_train_sm = sm.add_constant(X_train)
X_train_sm = X_train_sm.astype(float)
y_train = y_train.astype(float)  # Add this line to ensure y is also numeric

# Fit OLS model
ols_model = sm.OLS(y_train, X_train_sm).fit()

# Print statistical summary
print("\nOLS Regression Summary (statsmodels):")
print(ols_model.summary())

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
plt.xlabel("Actual Demand (MWh)")
plt.ylabel("Predicted Demand (MWh)")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Forecast Plot (Test)
plt.plot(dates_test, y_test, label='Test Data (Actual)', color='orange')
plt.plot(dates_test, y_test_pred, label='Forecast (Test)', color='red', linestyle='--')
plt.title("Daily Electricity Demand Forecast\n(Multiple Linear Regression Model)")
plt.xlabel("Date")
plt.ylabel("Demand (MWh)")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Forecast Plot (Train + Test)
plt.figure(figsize=(12, 6))
plt.plot(dates_train, y_train, label='Training Data (Actual)', color='blue')
plt.plot(dates_train, y_train_pred, label='Fitted Values (Train)', color='green', linestyle='--')
plt.plot(dates_test, y_test, label='Test Data (Actual)', color='orange')
plt.plot(dates_test, y_test_pred, label='Forecast (Test)', color='red', linestyle='--')
plt.title("Daily Electricity Demand Forecast\n(Multiple Linear Regression Model)")
plt.xlabel("Date")
plt.ylabel("Demand (MWh)")
plt.legend(loc='upper left')
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

# 5. Standardized Residuals vs Date
plt.figure(figsize=(10, 5))
plt.plot(dates_all, standardized_residuals_all, marker='o', linestyle='-', color='purple')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Date")
plt.ylabel("Standardized Residuals")
plt.title("Standardized Residuals vs Date")
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Histogram of Residuals
plt.figure(figsize=(5, 5))
plt.hist(residuals_all, bins=10, edgecolor='black', alpha=0.7)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Q-Q Plot of Residuals
plt.figure(figsize=(5, 5))
stats.probplot(residuals_all, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.grid(True)
plt.tight_layout()
plt.show()
