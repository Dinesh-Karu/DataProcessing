import pandas as pd

ref_file = "C:/_VAMK/_Coding/Data/Input/Ref Hourly Data Range 2020-2024.csv"
input_file = "C:/_VAMK/_Coding/Data/Input/Hourly Average temperature Vaasa 2020-2024.csv"
output_file = "C:/_VAMK/_Coding/Data/Output/Missing data points marked Hourly Avg Temp Vaasa 2020-2024.csv"

df = pd.read_csv(input_file, sep=",")
# Convert the DataFrame to a Dictionary
input_data_dict = df.to_dict(orient='records')

df = pd.read_csv(ref_file, sep=",")
output_data = []
i = 0

# Process each row in input data
for _, row in df.iterrows():
    ref_year = row["Year"]
    ref_month = row["Month"]
    ref_day = row["Day"]
    ref_hour = row["Hour"]

    if (ref_year == input_data_dict[i]["Year"] and
        ref_month == input_data_dict[i]["Month"] and
        ref_day == input_data_dict[i]["Day"] and
        ref_hour == int(input_data_dict[i]["Time"].split(":")[0])):
        
        output_data.append([ref_year, ref_month, ref_day, ref_hour, 
                            input_data_dict[i]["Avg_temperature"], 
                            input_data_dict[i]["Observation station"]])
        # if match then go to the next input data
        i += 1
    else:
        # otherwise input data is missing for the reference point
        output_data.append([ref_year, ref_month, ref_day, ref_hour, 
                            "missing", 
                            "missing"])

# Convert to DataFrame and Save to output file
output_df = pd.DataFrame(output_data, columns=["Year", "Month", "Day", "Hour", "Avg_temperature", "Observation_Station"])
output_df.to_csv(output_file, sep=",", index=False)

print(f"Processed data saved to {output_file}")