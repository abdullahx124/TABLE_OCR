import csv
import os
import pandas as pd
import glob
import os

def csv_file_creation(pdf_name, data):
    # Create the folder if it doesn't exist
    output_folder = f"../CSV_Output/{pdf_name}"
    os.makedirs(output_folder, exist_ok=True)

    # Assuming data is a list of dictionaries
    for idx, row_data in enumerate(data, start=1):
        filename = os.path.join(output_folder, f'output_{idx}.csv')
        with open(filename, 'w', newline='') as result_file:
            wr = csv.writer(result_file, dialect='excel')
            for values in row_data.values():
                wr.writerow(values)



# Specify the folder where CSV files are saved
    output_folder = f"../CSV_Output/{pdf_name}"

# Get a list of all CSV files in the output folder
    csv_files = glob.glob(os.path.join(output_folder, 'output_*.csv'))
    return csv_files

def csv_file_main(pdf_name, data):
    csv_files = csv_file_creation(pdf_name, data)

    # Read and display data from each CSV file
    for idx, csv_file in enumerate(csv_files, start=1):
        df = pd.read_csv(csv_file)
        print(f"Data from {csv_file}:")
        print(df.head(10))
        print("\n" + "-"*30 + f" End of DataFrame {idx} " + "-"*30 + "\n")