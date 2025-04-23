import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy.contrasts import Sum

df = pd.read_csv('C:/_VAMK/_Coding/Data/Input/Hourly Combined data Vaasa 2020-2024.csv')

categorical_cols = ['Month_No', 'Iso_Weekday', 'Hour', 'Sun_Flag']
for col in categorical_cols:
    df[col] = df[col].astype('category')

contrast = {
    'Month_No': Sum(), 'Iso_Weekday': Sum(), 'Hour': Sum(), 'Sun_Flag': Sum()
}

formula = 'Demand_MWh ~ C(Month_No, Sum) + C(Iso_Weekday, Sum) + C(Hour, Sum) + C(Sun_Flag, Sum)'
model = smf.ols(formula=formula, data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=3)

pd.set_option('display.float_format', '{:,.3f}'.format)
print(anova_table)
