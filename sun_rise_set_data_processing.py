import pandas as pd

input_file = "C:/_VAMK/_Coding/Data/Input/input.csv"
output_file = "C:/_VAMK/_Coding/Data/Output/output.csv"

df = pd.read_csv(input_file, sep=",")

output_data = []

# Process each row in input data
for _, row in df.iterrows():
    year = row["Year"]
    month = row["Month"]
    day = row["Day"]

    begin_hour = int(row["Sunrise"].split(":")[0])
    if int(row["Sunrise"].split(":")[1]) > 30:
        begin_hour += 1
    end_hour = int(row["Sunset"].split(":")[0])
    if int(row["Sunset"].split(":")[1]) > 30:
        end_hour += 1
    
    # Create 24-hour flag data
    for hour in range(1, 25):
        if begin_hour <= hour <= begin_hour + 1:
            flag = 2 
        elif begin_hour + 1 < hour < end_hour - 1:
            flag = 3
        elif end_hour - 1 <= hour <= end_hour:
            flag = 2
        else:
            flag = 1
        output_data.append([year, month, day, hour, flag])

# Convert to DataFrame
output_df = pd.DataFrame(output_data, columns=["Year", "Month", "Day", "Hour", "Flag"])
# Save to output file
output_df.to_csv(output_file, sep=",", index=False)

print(f"Processed data saved to {output_file}")