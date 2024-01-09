import pandas as pd

def preprocess_data(file_path):
    # Load DataFrame
    df = pd.read_csv(file_path)

    # Function to extract numeric part from 'HS Code'
    def extract_numeric(code):
        try:
            return int(''.join(filter(str.isdigit, code)))
        except ValueError:
            return 0

    # Drop unnecessary columns
    columns_to_drop = [
        "Unit", "Basic Duty (NTFN)", "Specific Duty (Rs)", "10% SWS",
        "IGST", "Basic Duty (SCH)", "Total Duty Specific",
        "Pref. Duty (A)", "Non Tariff Barriers", "Remark",
        "Unnamed: 15", "Unnamed: 16", "Unnamed: 17", "Unnamed: 18",
        "Import Policy ", "Item Description", "Level"
    ]

    # Check if each column in columns_to_drop is present in the DataFrame
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)

    # Assume your DataFrame is called df
    # Fill missing values and preprocess 'HS Code' as before
    df['HS Code'].fillna(value='', inplace=True)
    df['HS Code'] = df['HS Code'].str.replace(' ', '')
    df['HS Code'] = df['HS Code'].astype(str)
    df['HS Code'].replace('', '0', inplace=True)

    # Convert 'HS Code' to integer, handle multiple codes and special characters
    df['HS Code'] = df['HS Code'].apply(extract_numeric)

    # Convert the target column to numeric, handling errors as 'coerce'
    df['Total duty with SWS of 10% on BCD'] = pd.to_numeric(df['Total duty with SWS of 10% on BCD'], errors='coerce')

    # Drop rows with NaN values in the target variable
    df = df.dropna(subset=['Total duty with SWS of 10% on BCD']).reset_index(drop=True)

    X = df['HS Code']
    y = df['Total duty with SWS of 10% on BCD']

    return X, y

