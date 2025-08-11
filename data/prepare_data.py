import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from rdkit import Chem

# This file takes the 'extracted_table.csv' and splits the dataset into 2 - a training and validation. 
# i wonder about using a test set - there is not one in the model cross validation
#so part of me wonders if it is really necesaary - becuse i am trying to predict every
#single property with one module insteald one model per property. do this for now and then see

# normalisation code:

def _min_max_scale(series: pd.Series) -> pd.Series:
    """
    Applies Min-Max scaling to a Pandas Series.
    Scales values to a range between 0 and 1.

    Formula: X_normalized = (X - X_min) / (X_max - X_min)

    Args:
        series (pd.Series): The input Series to be scaled.

    Returns:
        pd.Series: The Min-Max scaled Series.
    """
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val: # Avoid division by zero if all values are the same
        return pd.Series(0.0, index=series.index)
    return (series - min_val) / (max_val - min_val)

def _z_score_scale(series: pd.Series) -> pd.Series:
    """
    Applies Z-score standardization to a Pandas Series.
    Transforms data to have a mean of 0 and a standard deviation of 1.

    Formula: X_normalized = (X - X_mean) / X_std

    Args:
        series (pd.Series): The input Series to be scaled.

    Returns:
        pd.Series: The Z-score standardized Series.
    """
    mean_val = series.mean()
    std_val = series.std()
    if std_val == 0: # Avoid division by zero if all values are the same
        return pd.Series(0.0, index=series.index)
    norm_series = (series - mean_val) / std_val
    return norm_series, mean_val, std_val

def normalize_dataframe_columns(df: pd.DataFrame, method: str = 'min_max') -> pd.DataFrame:
    """
    Normalizes all numerical columns in a Pandas DataFrame using the specified method.

    Args:
        df (pd.DataFrame): The input DataFrame containing numerical columns to be normalized.
        method (str): The normalization method to use.
                      Accepted values: 'min_max' (default) or 'z_score'.

    Returns:
        pd.DataFrame: A new DataFrame with the numerical columns normalized.
                      Non-numerical columns will remain unchanged.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_normalized = df.copy()

    # Identify numerical columns
    numerical_cols = df_normalized.select_dtypes(include=np.number).columns

    if numerical_cols.empty:
        print("No numerical columns found to normalize.")
        return df_normalized

    if method not in ['min_max', 'z_score']:
        raise ValueError("Invalid normalization method. Choose 'min_max' or 'z_score'.")
    property_stats = {}
    # Apply the selected scaling method to each numerical column
    for col in numerical_cols:
        property_stats[col] = {}
        if method == 'min_max':
            df_normalized[col] = _min_max_scale(df_normalized[col])
        elif method == 'z_score':
            df_normalized[col], mean_val, std_val = _z_score_scale(df_normalized[col])
            property_stats[col]['mean'] = mean_val
            property_stats[col]['std'] = std_val
    print('changed df nature in norm?', isinstance(df_normalized, pd.DataFrame))

    return df_normalized, property_stats

#cannonicalise smiles strings for consistent results:
def canonicalize_smiles(smiles):
    """
    Canonicalizes a SMILES string using RDKit.
    Returns None if the SMILES is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
        else:
            return None # Or you could return an empty string, or the original invalid SMILES
    except Exception:
        return None # Handle any other potential RDKit errors

def canonicalize_smiles_column(df: pd.DataFrame, smiles_column_name: str) -> pd.DataFrame:
    """
    Canonicalizes all SMILES strings in a specified DataFrame column in-place.

    Args:
        df (pd.DataFrame): The input DataFrame.
        smiles_column_name (str): The name of the column containing SMILES strings.

    Returns:
        pd.DataFrame: The DataFrame with the SMILES column canonicalized.
                      (Note: The operation is done in-place, but returning for chaining)
    """
    if smiles_column_name not in df.columns:
        print(f"Error: Column '{smiles_column_name}' not found in the DataFrame.")
        return df

    print(f"Canonicalizing SMILES in column '{smiles_column_name}'...")
    # Apply the canonicalization function to the specified column
    # .astype(str) is used to ensure all entries are treated as strings before passing to RDKit
    df[smiles_column_name] = df[smiles_column_name].astype(str).apply(canonicalize_smiles)
    print("Canonicalization complete.")
    return df

def rename_fully_extracted_solvent_types(df):
#CHANGED SOLVENT AMOUNTS, HERE IS ORIGIONAL: numbers = [22, 13, 16, 15, 18, 20, 8, 15, 6, 8, 10, 28, 8, 4]
#CATECHOL NUMBERS = [22, 13, 16, 15, 19, 21, 8, 15, 6, 8, 10, 29, 8, 4]
    SOLVENT_TYPES = ['alkane', 'aromatic', 'halohydrocarbon', 'ether', 'ketone', 'ester', 'nitrile', 'amine', 'amide', 'misc_N_compound', 'carboxylic_acid', 'monohydric_alcohol', 'polyhydric_alcohol', 'other']

    numbers = [22, 13, 16, 15, 18, 20, 8, 15, 6, 8, 10, 28, 8, 4]
    total = sum(numbers)
    if total != df.shape[0]:
        print(total), print(df.shape[0])
        raise ValueError("size mismatch")
    else:
        solvents_enumerated = []

        for i in range(len(numbers)):
            for j in range(numbers[i]):
                solvents_enumerated.append(SOLVENT_TYPES[i])

        df_solvent_types = df.drop(columns='solvent')
        df_solvent_types.insert(loc=0, column='solvent', value=solvents_enumerated)

    return df_solvent_types

# extract a random number of solvents for validation
def create_train_test_split(file_path, rename_solvents, test_percent):
    """
    Loads a CSV file and splits it into training and testing sets for missing value imputation.

    The function calculates a test set size equal to 10% of the total rows in the original
    dataset. It then attempts to sample this number of rows from the subset of data that
    contains no missing values.

    All rows with missing values are guaranteed to be in the training set.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        tuple: A tuple containing two pandas DataFrames:
               - train_df (pd.DataFrame): The training set.
               - test_df (pd.DataFrame): The test set (composed of complete rows).
    """
    try:
        # Load the entire dataset from the provided CSV file path
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path}. Total rows: {len(df)}")

        # Define the mapping for column renaming based on the corrected headers
        column_rename_map = {
            'solvent': 'solvent', # Assuming this is the 'solvent' column based on the example data
            'solvent smiles': 'SMILES',
            'ET(30)': 'ET30',
            'α': 'alpha',
            'β': 'beta',
            'π*': 'pi_star',
            'SA': 'SA',
            'SB': 'SB',
            'SP': 'SP',
            'SdP': 'SdP',
            'N (mol/cm3)': 'N_mol_cm3',
            'n' : 'n',
            'f(n)': 'fn',
            'δ': 'delta'

         # The last valid column header as per your clarification
        }
        # Filter the rename map to only include columns actually present in the DataFrame
        # This helps prevent errors if some old column names aren't found
        actual_rename_map = {old_name: new_name for old_name, new_name in column_rename_map.items() if old_name in df.columns}

        # Rename the columns
        if actual_rename_map:
            df = df.rename(columns=actual_rename_map)
            print(f"Renamed columns: {list(actual_rename_map.keys())} to {list(actual_rename_map.values())}")
        else:
            print("No specified columns found for renaming or no renaming needed.")

        if rename_solvents:
            df = rename_fully_extracted_solvent_types(df)
        print(f"Labeling solvent types.")
        df_norm, norm_stats_dict = normalize_dataframe_columns(df, method='z_score')
        print(f"Normalized columns.")

        df_canon = canonicalize_smiles_column(df_norm, 'SMILES')
        print(f"Canonicalized SMILES.")
        # Placeholder for your existing functions, assuming they are defined elsewhere
        # If these functions are not defined, this code will cause an error.
        # For demonstration, I'm commenting them out if you don't have them defined here.
        # You should uncomment them if they are part of your project.
        # df = rename_fully_extracted_solvent_types(df)
        # print(f"Labeling solvent types.")
        # df = normalize_dataframe_columns(df, method='z_score')
        # print(f"Normalized columns.")

        # Calculate the target size for the test set (10% of the original total)
        target_test_size = int(np.floor(test_percent * len(df_canon)))
        print(f"Target test set size (10% of total): {target_test_size} rows.")

        # Identify and isolate the rows that have no missing values at all.
        complete_rows_df = df_canon.dropna().copy()
        print(f"Found {len(complete_rows_df)} rows with no missing values.")

        # Check if there are enough complete rows to create the desired test set size
        if len(complete_rows_df) < target_test_size:
            print(f"Warning: Not enough complete rows ({len(complete_rows_df)}) to meet the target test size of {target_test_size}.")
            print("Using all available complete rows for the test set.")
            test_df = complete_rows_df.copy()
        else:
            # Randomly sample the target number of rows from the complete rows to create the test set.
            test_df = complete_rows_df.sample(n=target_test_size, random_state=42)

        print(f"Created test set with {len(test_df)} randomly sampled complete rows.")

        # The training set is created by dropping the rows that were selected for the test set.
        # This ensures all rows with missing values, plus the remaining complete rows, are in the training set.
        train_df = df_canon.drop(test_df.index)
        print(f"Created training set with {len(train_df)} rows.")

        return train_df, test_df, norm_stats_dict

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None, None
        