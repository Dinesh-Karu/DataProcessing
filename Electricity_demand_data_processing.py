import pandas as pd

input_file = "C:/_VAMK/_Coding/Data/Input/Hourly Electricity Demand Vaasa 2020-2024_Row.csv"
output_hourly_file = "C:/_VAMK/_Coding/Data/Output/Hourly Electricity Demand Vaasa 2020-2024.csv"
output_daily_file = "C:/_VAMK/_Coding/Data/Output/Daily Electricity Demand Vaasa 2020-2024.csv"
output_monthly_file = "C:/_VAMK/_Coding/Data/Output/Monthly Electricity Demand Vaasa 2020-2024.csv"
output_yearly_file = "C:/_VAMK/_Coding/Data/Output/Yearly Electricity Demand Vaasa 2020-2024.csv"

df = pd.read_csv(input_file, sep=",")
output_hourly_data = []
output_daily_data = []
output_monthly_data = []
output_yearly_data = []

prev_date = '0'
prev_month = '0'
day_total = 0
month_total = 0

# Process each row in input data
for _, row in df.iterrows():
    time_stamp = row["Timestamp"]
    date_ = time_stamp.split(" ")[0]
    time_ = time_stamp.split(" ")[1].split(":")[0]
    demand_ = row["Hourly measurements in kWh/h"]

    output_hourly_data.append([date_, time_, demand_])

    if prev_date != date_:
        month_ = date_.split(".")[1]
        ## day
        if prev_date != '0':
            output_daily_data.append([prev_date, day_total/1000])

            # month
            month_total += day_total
            if prev_month != month_:
                if prev_month != '0':
                    output_monthly_data.append([prev_date.split(".")[2], prev_month, month_total/1000])
                    month_total = 0
                else:
                    month_total = day_total

            day_total = demand_
        else:
            day_total = demand_
        prev_month = month_
    else:
        day_total += demand_

    prev_date = date_

# last day
output_daily_data.append([prev_date, day_total/1000])
# last month
month_total += day_total
output_monthly_data.append([prev_date.split(".")[2], prev_month, month_total/1000])

# Save hourly output file
output_df = pd.DataFrame(output_hourly_data, columns=["Date", "Time", "Demand(kWh)"])
output_df.to_csv(output_hourly_file, sep=",", index=False)
# Save daily output file
output_df = pd.DataFrame(output_daily_data, columns=["Date", "Demand(MWh)"])
output_df.to_csv(output_daily_file, sep=",", index=False)
# Save monthly output file
output_df = pd.DataFrame(output_monthly_data, columns=["Year", "Month", "Demand(MWh)"])
output_df.to_csv(output_monthly_file, sep=",", index=False)

print(f"Processed data saved to")
print(f"1. {output_hourly_file}")
print(f"2. {output_daily_file}")