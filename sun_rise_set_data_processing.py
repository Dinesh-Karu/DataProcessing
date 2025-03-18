import pandas as pd

# Read input file
input_file = "C:\_VAMK\_Coding\Data\Input\input.txt"
output_file = "C:\_VAMK\_Coding\Data\Output\output.txt"

# Load the input data
df = pd.read_csv(input_file, sep="\t")

# Initialize output list
output_data = []

# Process each row in input data
for _, row in df.iterrows():
    day = row["Day"]
    begin_hour = int(row["Begin"].split(":")[0])  # Extract hour
    end_hour = int(row["End"].split(":")[0])      # Extract hour
    
    # Create 24-hour flag data
    for hour in range(1, 25):
        flag = 1 if begin_hour <= hour <= end_hour else 0
        output_data.append([day, hour, flag])

# Convert to DataFrame
output_df = pd.DataFrame(output_data, columns=["Day", "Hour", "Flag"])

# Save to output file
output_df.to_csv(output_file, sep="\t", index=False)

print(f"Processed data saved to {output_file}")