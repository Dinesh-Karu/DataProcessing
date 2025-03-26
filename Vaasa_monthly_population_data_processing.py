import pandas as pd

input_file = "C:/_VAMK/_Coding/Data/Input/Population by month Vaasa.csv"
output_file = "C:/_VAMK/_Coding/Data/Output/Population by year Vaasa.csv.csv"

df = pd.read_csv(input_file, sep=",")
output_data = []
prev_year = '0'
annual_population = 0

# Process each row in input data
for _, row in df.iterrows():
    year_month = row["Year_Month"]
    population = row["Population"]

    year = row["Year_Month"].split("M")[0]
    if prev_year != year:
        if prev_year != '0':
            output_data.append([year, annual_population])
        annual_population = population
    else:
        annual_population += population

    prev_year = year

# Convert to DataFrame
output_df = pd.DataFrame(output_data, columns=["Year", "Annual_Population"])
# Save to output file
output_df.to_csv(output_file, sep=",", index=False)

print(f"Processed data saved to {output_file}")