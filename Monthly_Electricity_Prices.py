import pandas as pd

input_file = "C:/_VAMK/_Coding/Data/Input/Monthly Price of electricity by type of consumer Finland 2020-2024.csv"
output_file = "C:/_VAMK/_Coding/Data/Output/Monthly Avg Price of electricity Finland 2020-2024.csv"

df = pd.read_csv(input_file, sep=",")
output_data = []
prev_year_month = '0'
total = 0
count = 0

# Process each row in input data
for _, row in df.iterrows():
    price = row["Price(c/kWh)"]
    year_month = row["Year_Month"]
    price_component = row["Price component"]

    if prev_year_month != year_month:
        if prev_year_month != '0':
            output_data.append([prev_year_month.split("M")[0], prev_year_month.split("M")[1], round(total/count, 3)])
            total = 0
            count = 0
    elif price_component == "Total price":
        total += price
        count += 1

    prev_year_month = year_month

# last month average
output_data.append([prev_year_month.split("M")[0], prev_year_month.split("M")[1], round(total/count, 3)])


# Convert to DataFrame
output_df = pd.DataFrame(output_data, columns=["Year", "Month", "Avg_Electricity_Price"])
# Save to output file
output_df.to_csv(output_file, sep=",", index=False)

print(f"Processed data saved to {output_file}")