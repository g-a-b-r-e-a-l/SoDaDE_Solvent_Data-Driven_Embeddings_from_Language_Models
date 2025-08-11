import pandas as pd
import numpy as np
import random
from rdkit import Chem

# ----------------- Normalisation Helpers -----------------
def normalize_with_stats(df: pd.DataFrame, stats: dict, method: str = 'z_score') -> pd.DataFrame:
    """Apply precomputed normalization stats to a dataframe."""
    df_norm = df.copy()
    numerical_cols = df_norm.select_dtypes(include=np.number).columns

    for col in numerical_cols:
        if col in stats:
            if method == 'min_max':
                min_val = stats[col]['min']
                max_val = stats[col]['max']
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val) if max_val != min_val else 0.0
            elif method == 'z_score':
                mean_val = stats[col]['mean']
                std_val = stats[col]['std']
                df_norm[col] = (df_norm[col] - mean_val) / std_val if std_val != 0 else 0.0
    return df_norm

def compute_normalization_stats(df: pd.DataFrame, method: str = 'z_score') -> dict:
    """Compute normalization parameters from dataframe."""
    stats = {}
    numerical_cols = df.select_dtypes(include=np.number).columns

    for col in numerical_cols:
        if method == 'min_max':
            stats[col] = {'min': df[col].min(), 'max': df[col].max()}
        elif method == 'z_score':
            stats[col] = {'mean': df[col].mean(), 'std': df[col].std()}
    return stats

# ----------------- SMILES Canonicalisation -----------------
def canonicalize_smiles(smiles):
    """Canonicalizes a SMILES string using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
        else:
            return None
    except Exception:
        return None

def canonicalize_smiles_column(df: pd.DataFrame, smiles_column_name: str) -> pd.DataFrame:
    """Canonicalizes all SMILES strings in a specified DataFrame column."""
    if smiles_column_name not in df.columns:
        print(f"Error: Column '{smiles_column_name}' not found in the DataFrame.")
        return df
    print(f"Canonicalizing SMILES in column '{smiles_column_name}'...")
    df[smiles_column_name] = df[smiles_column_name].astype(str).apply(canonicalize_smiles)
    print("Canonicalization complete.")
    return df

def rename_columns(df):
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
            'δ': 'delta'}
    
    actual_rename_map = {old_name: new_name for old_name, new_name in column_rename_map.items() if old_name in df.columns}

    # Rename the columns
    if actual_rename_map:
        df = df.rename(columns=actual_rename_map)
        print(f"Renamed columns: {list(actual_rename_map.keys())} to {list(actual_rename_map.values())}")
    else:
        print("No specified columns found for renaming or no renaming needed.")

    return df


# ----------------- Solvent Naming -----------------
def rename_fully_extracted_solvent_types(df):
    SOLVENT_TYPES = [
        'alkane', 'aromatic', 'halohydrocarbon', 'ether', 'ketone', 'ester',
        'nitrile', 'amine', 'amide', 'misc_N_compound', 'carboxylic_acid',
        'monohydric_alcohol', 'polyhydric_alcohol', 'other'
    ]
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

# ----------------- Individual Solvent Selection Helper -----------------
def select_individual_test_solvents(df, n_solvents=5, num_solvents_per_type=1):
    """
    Select individual solvents (by SMILES) from specified solvent types for testing.
    
    Args:
        df: DataFrame with 'solvent' and 'SMILES' columns
        test_solvent_types: List of solvent types to select from. If None, randomly pick 2.
        num_solvents_per_type: Number of individual solvents to select from each type
    
    Returns:
        list: SMILES strings of selected test solvents
    """

    solvent_types = df['solvent'].unique().tolist()

    if n_solvents < len(solvent_types):
        print("Issue - number of solvent types requested is too much")
    test_solvent_types = random.sample(solvent_types, n_solvents)
    print(f"Randomly selected solvent types for test set: {test_solvent_types}")
    
    test_smiles = []
    
    for solvent_type in test_solvent_types:
        # Get all unique SMILES for this solvent type
        type_solvents = df[df['solvent'] == solvent_type]['SMILES'].unique().tolist()
        # Remove None values if any
        type_solvents = [smiles for smiles in type_solvents if smiles is not None]
        
        if len(type_solvents) < num_solvents_per_type:
            print(f"Warning: Only {len(type_solvents)} solvents available in {solvent_type}, "
                  f"but {num_solvents_per_type} requested. Using all available.")
            selected = type_solvents
        else:
            selected = random.sample(type_solvents, num_solvents_per_type)
        
        test_smiles.extend(selected)
        print(f"Selected from {solvent_type}: {selected}")
    
    return test_smiles

# ----------------- Main Split Function -----------------
def create_train_val_test_split(file_path, rename_solvents=True, method='z_score',
                                n_solvents=5, val_percent=0.2):
    """
    Creates train, validation, and test sets where the test set is made of individual
    solvents (by SMILES) from specified solvent types BEFORE normalization.
    Training and validation sets include NaN sequences, while test set contains only
    complete sequences (no NaN values).
    
    Args:
        file_path: Path to CSV file
        rename_solvents: Whether to rename solvent types
        method: Normalization method ('z_score' or 'min_max')
        test_solvent_types: List of solvent types to select test solvents from. 
                           If None, randomly selects 2 types.
        val_percent: Percentage of full dataset to use for validation
    
    Returns:
        train_df_norm: Normalized training set (includes NaN sequences)
        val_df_norm: Normalized validation set (includes NaN sequences)
        test_df_norm: Normalized test set (no NaN values)
        norm_stats: Normalization statistics
        test_solvents_info: Dictionary with test solvent types and SMILES
        
    Raises:
        ValueError: If there are insufficient samples for validation set after test extraction
    """
    df = pd.read_csv(file_path)
    df = rename_columns(df)
    print('renamed columns to ', df.columns )
    
    # Rename solvents if requested
    if rename_solvents:
        df = rename_fully_extracted_solvent_types(df)
    print('created solvent types')
    
    # Canonicalize SMILES
    df = canonicalize_smiles_column(df, 'SMILES')
    print('canonicalised SMILES')
    
    # Create clean dataset for test/val extraction (no NaN values)
    df_clean = df.dropna()
    print(f"Clean dataset size: {len(df_clean)} samples (from original {len(df)})")
    
    # Calculate target sizes based on full dataset
    total_size = len(df)
    target_val_size = int(total_size * val_percent)
    
    # Select individual test solvents from clean dataset only
    test_smiles = select_individual_test_solvents(df_clean, n_solvents)
    
    # Step 1 — Extract test set from clean dataset
    test_df = df_clean[df_clean['SMILES'].isin(test_smiles)].copy()
    
    print(f"Test set size: {len(test_df)} samples (no NaN)")
    
    # Step 2 — Extract validation set from full dataset (excluding test SMILES)
    remaining_full_df = df[~df['SMILES'].isin(test_smiles)].copy()
    
    # Check if we have enough samples for validation
    if len(remaining_full_df) < target_val_size:
        raise ValueError(f"Not enough samples for validation set. "
                        f"Need {target_val_size} samples but only have {len(remaining_full_df)} "
                        f"samples remaining after test set extraction.")
    
    val_df = remaining_full_df.sample(n=target_val_size, random_state=42)
    
    # Step 3 — Create training set from remaining data (includes NaN sequences)
    test_val_smiles = set(test_df['SMILES'].tolist() + val_df['SMILES'].tolist())
    train_df = df[~df['SMILES'].isin(test_val_smiles)].copy()
    print(f'Train set size: {len(train_df)} (includes NaN), Val set size: {len(val_df)} (includes NaN), Test set size: {len(test_df)} (no NaN)')
    print(f'Validation percentage of original dataset: {target_val_size/total_size:.2%}')
    print(f'NaN sequences in training set: {train_df.isna().any(axis=1).sum()}')
    print(f'NaN sequences in validation set: {val_df.isna().any(axis=1).sum()}')
    
    # Step 3 — Compute stats from training set only
    norm_stats = compute_normalization_stats(train_df, method)
    print('extracting norm stats')
    
    # Step 4 — Apply normalization to all sets
    train_df_norm = normalize_with_stats(train_df, norm_stats, method)
    val_df_norm = normalize_with_stats(val_df, norm_stats, method)
    test_df_norm = normalize_with_stats(test_df, norm_stats, method)
    
    # Create info about test solvents
    test_solvents_info = {
        'test_solvent_types': test_df['solvent'].unique().tolist(),
        'test_smiles': test_smiles,
        'test_solvent_details': test_df[['solvent', 'SMILES']].drop_duplicates().to_dict('records')
    }

    return train_df_norm, val_df_norm, test_df_norm, norm_stats, test_solvents_info