import pandas as pd
from datetime import date

input_file = "C:/_VAMK/_Coding/Data/Input/Daily Sun in Vaasa 2024.csv"
output_file = "C:/_VAMK/_Coding/Data/Output/Hourly Sun in Vaasa 2024.csv"

df = pd.read_csv(input_file, sep=",")

output_data = []

# Process each row in input data
for _, row in df.iterrows():
    year = row["Year"]
    month = row["Month"]
    day = row["Day"]
    isoweekday = date(int(year), int(month), int(day)).isoweekday()

    begin_hour = int(row["Sunrise"].split(":")[0])
    if int(row["Sunrise"].split(":")[1]) > 30:  ##if minutes part is more than 30
        begin_hour += 1
    end_hour = int(row["Sunset"].split(":")[0]) 
    if int(row["Sunset"].split(":")[1]) > 30:   ##if minutes part is more than 30
        end_hour += 1
    
    # Create 24-hour flag data
    for hour in range(0, 24):
        if begin_hour <= hour <= begin_hour + 1:
            flag = 1 
        elif begin_hour + 1 < hour < end_hour - 1:
            flag = 2
        elif end_hour - 1 <= hour <= end_hour:
            flag = 1
        else:
            flag = 0
        output_data.append([year, month, day, isoweekday, hour, flag])

# Convert to DataFrame
output_df = pd.DataFrame(output_data, columns=["Year", "Month", "Day", "Iso_Weekday", "Hour", "Sun_Flag"])
# Save to output file
output_df.to_csv(output_file, sep=",", index=False)

print(f"Processed data saved to {output_file}")