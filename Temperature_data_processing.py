import pandas as pd

input_file = "C:/_VAMK/_Coding/Data/Input/Hourly Avg Temp Vaasa 2020-2024 (missing values fixed).csv"
output_daily_file = "C:/_VAMK/_Coding/Data/Output/Daily Avg Temp Vaasa 2020-2024.csv"
output_monthly_file = "C:/_VAMK/_Coding/Data/Output/Monthly Avg Temp Vaasa 2020-2024.csv"

df = pd.read_csv(input_file, sep=",")
output_daily_data = []
output_monthly_data = []

prev_date = '0'
prev_month = '0'
prev_year = '0'
day_total = 0
month_total = 0
year_total = 0
no_of_days = 0
day_min = 100
day_max = -100
month_min = 100
month_max = -100

# Process each row in input data
for _, row in df.iterrows():
    year_ = row["Year"]
    month_ = row["Month"]
    date_ = row["Day"]
    hour_ = row["Hour"]
    temp_ = row["Avg_temperature"]

    ## day
    if prev_date != date_:
        if prev_date != '0':
            day_avg = day_total/24
            no_of_days += 1
            if month_min > day_min:
                month_min = day_min
            if month_max < day_max:
                month_max = day_max
            output_daily_data.append([year_, prev_month, prev_date, day_avg, day_min, day_max])

            # month
            if prev_month != month_:
                if prev_month != '0':
                    month_total += day_avg
                    output_monthly_data.append([year_, prev_month, month_total/no_of_days, month_min, month_max])
                    no_of_days = 0

                month_total = 0
                month_min = temp_
                month_max = temp_
            else:
                month_total += day_avg

        day_total = temp_
        day_min = temp_
        day_max = temp_
    else:
        day_total += temp_
        if day_min > temp_:
            day_min = temp_
        if day_max < temp_:
            day_max = temp_
    
    prev_date = date_
    prev_month = month_

# last day
day_avg = day_total/24
output_daily_data.append([year_, month_, prev_date, day_avg, day_min, day_max])

# last month
if month_min > day_min:
    month_min = day_min
if month_max < day_max:
    month_max = day_max
no_of_days += 1
month_total += day_avg
output_monthly_data.append([year_, prev_month, month_total/no_of_days, month_min, month_max])

# Save daily output file
output_df = pd.DataFrame(output_daily_data, columns=["Year", "Month", "Day", "Avg_temperature", "Min_temperature", "Max_temperature"])
output_df.to_csv(output_daily_file, sep=",", index=False)
# Save monthly output file
output_df = pd.DataFrame(output_monthly_data, columns=["Year", "Month", "Avg_temperature", "Min_temperature", "Max_temperature"])
output_df.to_csv(output_monthly_file, sep=",", index=False)

print(f"Processed data saved to")
print(f"1. {output_daily_file}")
print(f"2. {output_monthly_file}")
