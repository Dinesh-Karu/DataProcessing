import pandas as pd

input_file = "C:/_VAMK/_Coding/Data/Input/All automobiles Electricity Ostrobothnia.csv"
output_file = "C:/_VAMK/_Coding/Data/Output/EVs by year Vaasa.csv.csv"

df = pd.read_csv(input_file, sep=",")
output_data = []
prev_year = '0'
total = 0

# Process each row in input data
for _, row in df.iterrows():
    ev_count = row["EV_Count"]

    year = row["Year_Quarter"].split("Q")[0]
    if prev_year != year:
        if prev_year != '0':
            output_data.append([year, round(total/4)])
        total = ev_count
    else:
        total += ev_count

    prev_year = year

# Convert to DataFrame
output_df = pd.DataFrame(output_data, columns=["Year", "Avg_EVs"])
# Save to output file
output_df.to_csv(output_file, sep=",", index=False)

print(f"Processed data saved to {output_file}")
