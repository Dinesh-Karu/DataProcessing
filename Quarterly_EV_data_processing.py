import pandas as pd

ref_file = "C:/_VAMK/_Coding/Data/Input/Ref Hourly Data Range 2020-2024.csv"
input_file = "C:/_VAMK/_Coding/Data/Input/Quarterly EVs Ostrobothnia 2020-2024.csv"
output_hourly_file = "C:/_VAMK/_Coding/Data/Output/Hourly EVs Ostrobothnia 2020-2024.csv"
output_daily_file = "C:/_VAMK/_Coding/Data/Output/Daily EVs Ostrobothnia 2020-2024.csv"
output_monthly_file = "C:/_VAMK/_Coding/Data/Output/Monthly EVs Ostrobothnia 2020-2024.csv"

df = pd.read_csv(input_file, sep=",")
# Convert the DataFrame to a Dictionary
input_data_dict = df.to_dict(orient='records')

df = pd.read_csv(ref_file, sep=",")
output_hourly_data = []
output_daily_data = []
output_monthly_data = []
ref_prev_month = -1
i = 0

# Process each row in input data
for _, row in df.iterrows():
    ref_year = row["Year"]
    ref_month = int(row["Month"])
    ref_day = int(row["Day"])
    ref_hour = row["Hour"]

    if ref_month != ref_prev_month and ref_prev_month%3 == 0:
        i += 1

    output_hourly_data.append([ref_year, ref_month, ref_day, ref_hour, input_data_dict[i]["EV_Count"]])

    if ref_hour == 0:
        output_daily_data.append([ref_year, ref_month, ref_day, input_data_dict[i]["EV_Count"]])

    if ref_prev_month == -1 or ref_month != ref_prev_month:
        output_monthly_data.append([ref_year, ref_month, input_data_dict[i]["EV_Count"]])

    ref_prev_month = ref_month

# Convert to DataFrame and Save to output file
output_df = pd.DataFrame(output_hourly_data, columns=["Year", "Month", "Day", "Hour", "EV_Count"])
output_df.to_csv(output_hourly_file, sep=",", index=False)
output_df = pd.DataFrame(output_daily_data, columns=["Year", "Month", "Day", "EV_Count"])
output_df.to_csv(output_daily_file, sep=",", index=False)
output_df = pd.DataFrame(output_monthly_data, columns=["Year", "Month", "EV_Count"])
output_df.to_csv(output_monthly_file, sep=",", index=False)

print(f"Processed data saved to")
print(f"1. {output_hourly_file}")
print(f"2. {output_daily_file}")
print(f"3. {output_monthly_file}")