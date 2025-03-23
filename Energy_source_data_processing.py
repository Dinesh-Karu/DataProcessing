import pandas as pd

input_file = "C:/_VAMK/_Coding/Data/Input/Total energy consumption by energy source 2010Q1-2024Q3.csv"
output_file = "C:/_VAMK/_Coding/Data/Output/Energy_source_output.csv"

df = pd.read_csv(input_file, sep=",")

output_data = []
qty_array = []
col_heading = []
i = 0

# Process each row in input data
for _, row in df.iterrows():
    i += 1
    qty_array.append(row["Quantity (GWh)"])

    # extract column heading from first 15 rows
    if (i <= 15):
        col_heading.append(row["Energy source"])

    # 15 rows will represent as 1 row with 15 columns in the output file
    if (i%15 == 0):
        quarter = row["Quarter"]
        output_data.append([quarter, qty_array[i-15], qty_array[i-14], qty_array[i-13], qty_array[i-12], qty_array[i-11], qty_array[i-10], qty_array[i-9], qty_array[i-8], qty_array[i-7], qty_array[i-6], qty_array[i-5], qty_array[i-4], qty_array[i-3], qty_array[i-2], qty_array[i-1]])


# Convert to DataFrame
output_df = pd.DataFrame(output_data, columns=["Quarter",
                                               col_heading[0], #"SSS TOTAL ENERGY CONSUMPTION", 
                                               col_heading[1], #"1 Renewable energy", 
                                               col_heading[2], #"1.1 Hydro power", 
                                               col_heading[3], #"1.2 Wind power", 
                                               col_heading[4], #"1.3 Wood fuels",
                                               col_heading[5], #"1.4 Other renewable energy",
                                               col_heading[6], #"2 Fossil fuels and peat",
                                               col_heading[7], #"2.1 Oil (fossil)",
                                               col_heading[8], #"2.2 Coal",
                                               col_heading[9], #"2.3 Natural gas",
                                               col_heading[10], #"2.4 Peat",
                                               col_heading[11], #"2.5 Other fossil fuels",
                                               col_heading[12], #"3 Nuclear energy",
                                               col_heading[13], #"4 Net imports of electricity",
                                               col_heading[14],]) #"5 Others"])
# Save to output file
output_df.to_csv(output_file, sep=",", index=False)

print(f"Processed data saved to {output_file}")