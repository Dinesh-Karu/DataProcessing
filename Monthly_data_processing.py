import pandas as pd

ref_file = "C:/_VAMK/_Coding/Data/Input/Ref Hourly Data Range 2020-2024.csv"
input_file1 = "C:/_VAMK/_Coding/Data/Input/Monthly Avg Price of electricity Finland 2020-2024.csv"
output_hourly_file1 = "C:/_VAMK/_Coding/Data/Output/Hourly Avg Price of electricity Finland 2020-2024.csv"
output_daily_file1 = "C:/_VAMK/_Coding/Data/Output/Daily Avg Price of electricity Finland 2020-2024.csv"
input_file2 = "C:/_VAMK/_Coding/Data/Input/Monthly Population in Vaasa 2020-2024.csv"
output_hourly_file2 = "C:/_VAMK/_Coding/Data/Output/Hourly Population in Vaasa 2020-2024.csv"
output_daily_file2 = "C:/_VAMK/_Coding/Data/Output/Daily Population in Vaasa 2020-2024.csv"

df = pd.read_csv(input_file1, sep=",")
input_data_dict1 = df.to_dict(orient='records')
df = pd.read_csv(input_file2, sep=",")
input_data_dict2 = df.to_dict(orient='records')

df = pd.read_csv(ref_file, sep=",")
output_hourly_data1 = []
output_daily_data1 = []
output_hourly_data2 = []
output_daily_data2 = []
ref_prev_month = -1
i = 0

# Process each row in input data
for _, row in df.iterrows():
    ref_year = row["Year"]
    ref_month = int(row["Month"])
    ref_day = int(row["Day"])
    ref_hour = row["Hour"]

    if ref_month != ref_prev_month and ref_prev_month != -1:
        i += 1

    output_hourly_data1.append([ref_year, ref_month, ref_day, ref_hour, input_data_dict1[i]["Avg_Electricity_Price"]])
    output_hourly_data2.append([ref_year, ref_month, ref_day, ref_hour, input_data_dict2[i]["Population"]])

    if ref_hour == 0:
        output_daily_data1.append([ref_year, ref_month, ref_day, input_data_dict1[i]["Avg_Electricity_Price"]])
        output_daily_data2.append([ref_year, ref_month, ref_day, input_data_dict2[i]["Population"]])

    ref_prev_month = ref_month

# Convert to DataFrame and Save to output file
output_df = pd.DataFrame(output_hourly_data1, columns=["Year", "Month", "Day", "Hour", "Avg_Electricity_Price"])
output_df.to_csv(output_hourly_file1, sep=",", index=False)
output_df = pd.DataFrame(output_daily_data1, columns=["Year", "Month", "Day", "Avg_Electricity_Price"])
output_df.to_csv(output_daily_file1, sep=",", index=False)
output_df = pd.DataFrame(output_hourly_data2, columns=["Year", "Month", "Day", "Hour", "Population"])
output_df.to_csv(output_hourly_file2, sep=",", index=False)
output_df = pd.DataFrame(output_daily_data2, columns=["Year", "Month", "Day", "Population"])
output_df.to_csv(output_daily_file2, sep=",", index=False)

print(f"Processed data saved to")
print(f"1. {output_hourly_file1}")
print(f"2. {output_daily_file1}")
print(f"3. {output_hourly_file1}")
print(f"4. {output_daily_file1}")
