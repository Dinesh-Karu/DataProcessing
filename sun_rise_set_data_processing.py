import pandas as pd
from datetime import date

input_file = "C:/_VAMK/_Coding/Data/Input/Daily Sun in Vaasa 2020-2024.csv"
output_hourly_file = "C:/_VAMK/_Coding/Data/Output/Hourly Sun in Vaasa 2020-2024.csv"
output_daily_file = "C:/_VAMK/_Coding/Data/Output/Daily Sun duration in Vaasa 2020-2024.csv"
output_monthly_file = "C:/_VAMK/_Coding/Data/Output/Monthly Sun duration in Vaasa 2020-2024.csv"

df = pd.read_csv(input_file, sep=",")

output_hourly_data = []
output_daily_data = []
output_monthly_data = []
# Dictionay with days labeled
weekdays_ = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
# Dictionay with months labeled
months_ = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",	5: "May", 6: "Jun",	7: "Jul", 8: "Aug",	9: "Sep", 10: "Oct", 11: "Nov",	12: "Dec"}
prev_month = '0'
month_total = 0

# Process each row in input data
for _, row in df.iterrows():
    year_ = row["Year"]
    month_ = row["Month"]
    day_ = row["Day"]
    duration_ = float(row["Length"].split(":")[0] + "." + row["Length"].split(":")[1])
    iso_weekday = date(int(year_), int(month_), int(day_)).isoweekday()

    output_daily_data.append([year_, month_, months_[month_], day_, iso_weekday, weekdays_[iso_weekday], duration_])

    if prev_month != month_:
        if prev_month != '0':
            output_monthly_data.append([year_, prev_month, months_[prev_month], month_total])
        month_total = duration_
    else:
        month_total += duration_
    prev_month = month_

    begin_hour = int(row["Sunrise"].split(":")[0])
    if int(row["Sunrise"].split(":")[1]) > 30:  ##if minutes part is more than 30
        begin_hour += 1
    end_hour = int(row["Sunset"].split(":")[0]) 
    if int(row["Sunset"].split(":")[1]) > 30:   ##if minutes part is more than 30
        end_hour += 1
    
    # Create 24-hour flag data
    for hour_ in range(0, 24):
        if begin_hour <= hour_ <= begin_hour + 1:
            flag_ = 1 
        elif begin_hour + 1 < hour_ < end_hour - 1:
            flag_ = 2
        elif end_hour - 1 <= hour_ <= end_hour:
            flag_ = 1
        else:
            flag_ = 0
        output_hourly_data.append([year_, month_, months_[month_], day_, iso_weekday, weekdays_[iso_weekday], hour_, flag_])

# last month
output_monthly_data.append([year_, prev_month, months_[prev_month], month_total])

# Save hourly output
output_df = pd.DataFrame(output_hourly_data, columns=["Year", "Month_No", "Month", "Day", "Iso_Weekday", "Day of the Week", "Hour", "Sun_Flag"])
output_df.to_csv(output_hourly_file, sep=",", index=False)
# Save daily output
output_df = pd.DataFrame(output_daily_data, columns=["Year", "Month_No", "Month", "Day", "Iso_Weekday", "Day of the Week", "Duration"])
output_df.to_csv(output_daily_file, sep=",", index=False)
# Save monthly output
output_df = pd.DataFrame(output_monthly_data, columns=["Year", "Month_No", "Month", "Duration"])
output_df.to_csv(output_monthly_file, sep=",", index=False)

print(f"Processed data saved to")
print(f"1. {output_hourly_file}")
print(f"2. {output_daily_file}")
print(f"3. {output_monthly_file}")