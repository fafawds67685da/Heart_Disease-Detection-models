import pandas as pd

# Specify the path to your CSV file
csv_file_path = r"C:\Users\Hp\Downloads\heart.csv"

# Read the first 825 rows of the CSV file into a DataFrame
df = pd.read_csv(csv_file_path, nrows=825)

# Display the first few rows of the DataFrame to confirm
print(df)

# Optionally, display the total number of rows read
print(f"Total rows read: {len(df)}")