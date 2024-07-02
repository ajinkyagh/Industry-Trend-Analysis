import pandas as pd

file_path = 'organizations-10000.csv'    # Load the CSV file
data = pd.read_csv(file_path)

print(data.head())          # Display the first few rows of the dataset to understand its structure and contents

# Data Cleaning and Transformation Steps
missing_values = data.isnull().sum()    # Check for missing values

duplicate_rows = data.duplicated().sum()   # Check for duplicate rows

# Standardize and clean the 'Country' and 'Website' columns for consistency
data['Country'] = data['Country'].str.title()   # Capitalize country names
data['Website'] = data['Website'].str.lower()   # Ensure website URLs are in lowercase

# Data Profiling Information
profile_info = {
    'Column': data.columns,
    'Non-Null Count': data.notnull().sum(),
    'Data Type': data.dtypes,
    'Unique Values': data.nunique(),
    'Missing Values': missing_values,
}

# Convert profile_info dictionary to a DataFrame for better readability
profile_info_df = pd.DataFrame(profile_info)

# Removing duplicate rows if any
data_cleaned = data.drop_duplicates()

# Summary of changes
changes_summary = {
    'Original Rows': len(data),
    'Cleaned Rows': len(data_cleaned),
    'Duplicate Rows Removed': duplicate_rows,
}

print(profile_info_df)
print(changes_summary)

