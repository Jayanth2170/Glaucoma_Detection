import pandas as pd
import os

# Path to your existing CSV file (directly under your project)
csv_file_path = 'training_losses.csv'  # Update this if your CSV file is named differently
# Path where you want to save the Excel file
excel_file_path = 'training_losses.xlsx'  # You can change the name if desired

# Check if the CSV file exists
if os.path.exists(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Save the DataFrame to an Excel file
    df.to_excel(excel_file_path, index=False)  # index=False to avoid writing row indices

    print(f"Data successfully exported to {excel_file_path}")
else:
    print(f"CSV file not found at: {csv_file_path}")
