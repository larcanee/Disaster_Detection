import pandas as pd
import os

def combine_tsv_to_csv(tsv_folder, csv_output_file):
    # Get a list of all TSV files in the specified folder
    tsv_files = [f for f in os.listdir(tsv_folder) if f.endswith('.tsv')]

    # Initialize an empty list to store DataFrames
    dataframes = []

    # Loop through each TSV file and append its data to the list
    for tsv_file in tsv_files:
        tsv_path = os.path.join(tsv_folder, tsv_file)
        df = pd.read_csv(tsv_path, sep='\t')
        dataframes.append(df)

    # Concatenate all DataFrames in the list
    combined_data = pd.concat(dataframes, ignore_index=True)

    # Save the combined data to a CSV file
    combined_data.to_csv(csv_output_file, index=False)
    print(f"Combined data saved to {csv_output_file}")

# Specify the folder containing TSV files and the output CSV file
tsv_folder_path = 'Testfiles'
output_csv_path = 'categories_text.csv'

# Call the function to combine TSV files into a CSV file
combine_tsv_to_csv(tsv_folder_path, output_csv_path)
