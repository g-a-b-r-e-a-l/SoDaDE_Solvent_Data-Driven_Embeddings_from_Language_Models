import pandas as pd
import torch

from config import WORD_TOKENS, TOKEN_TYPES, VALS_COLUMNS, WORD_COLUMNS

def load_and_clean_csv(DATA_PATH):
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Data loaded successfully from {DATA_PATH}. Shape: {df.shape}")

        df_clean = df.dropna(subset=['SMILES']).copy()
        print(f"DataFrame after removing rows with missing SMILES: {df_clean.shape}")

        print("First 5 rows:")
        print(df.head())

        return df_clean

    except FileNotFoundError:
        print(f"Error: The file '{DATA_PATH}' was not found. Please ensure it's in the correct directory or update DATA_PATH.")

def create_values_tensor(df, max_len_plus_2, VALS_COLUMNS):
    values_tensor = torch.full((df.shape[0], max_len_plus_2), -100, dtype=torch.float)


    col_id = 1 # account for start token column
    for i in df.columns:
        if i in VALS_COLUMNS:
            values_tensor[:, col_id] = torch.tensor(df[i])
        col_id += 1

    missing_val_mask = torch.isnan(values_tensor)

    return values_tensor, missing_val_mask

def create_diff_token_vocabs(WORD_TOKENS, TOKEN_TYPES):
    word_vocab = {col: i for i, col in enumerate(WORD_TOKENS)}
    VOCAB_SIZE_COLUMNS = len(word_vocab)
    token_type_vocab = {token_type: i for i, token_type in enumerate(TOKEN_TYPES)}

    print(f"Column Vocabulary: {word_vocab}")
    print(f"Token Type Vocabulary: {token_type_vocab}")


    return token_type_vocab, word_vocab, VOCAB_SIZE_COLUMNS


def word_token_indicies(df, max_len_plus_2, word_columns, word_vocab):
    word_index_tensor = torch.zeros((df.shape[0], max_len_plus_2), dtype=torch.long)
    col_id = 1 # account for start token column
    for i in df.columns:
        if i in WORD_COLUMNS:
            word_index_tensor[:, col_id] = torch.tensor(df[i].map(word_vocab))
        col_id += 1
    return word_index_tensor

def create_fingerprints(df, max_len_plus_2, smiles_column, tokenizer, model_chemberta):
    smiles_list = df[smiles_column].tolist()
    SMILES_fps = torch.zeros((df.shape[0], max_len_plus_2, CHEMBERTA_FP_DIM), dtype=torch.float)

    # Tokenize SMILES strings

    smiles_tokenized_inputs = tokenizer(smiles_list, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        # Move inputs to device if you have a GPU for model_chemberta
        # smiles_tokenized_inputs = {k: v.to(device) for k, v in smiles_tokenized_inputs.items()}
        outputs = model_chemberta(**smiles_tokenized_inputs, output_hidden_states=True)
        smiles_chemberta_fps = outputs.hidden_states[-1].mean(dim=1) # Shape: (batch_size, CHEMBE`RTA_FP_DIM)

    SMILES_fps[:, 2, :] = smiles_chemberta_fps

    return SMILES_fps



def create_token_type_tensor(df, max_len_plus_2, WORD_COLUMNS, VALS_COLUMNS, SMILES_COLUMN, token_type_vocab):
    tensor_dict = {}

    for i in list(token_type_vocab.keys()):
        tensor_dict[i] = torch.full((df.shape[0], ), token_type_vocab[i], dtype=torch.int)

    token_type_tensor = torch.zeros(df.shape[0], max_len_plus_2, dtype=torch.int)

    token_type_tensor[:, 0] = tensor_dict['CLS_TOKEN']
    token_type_tensor[:, -1] = tensor_dict['SEP_TOKEN']
    tensor_index = 0
    for i in range(df.shape[1]):
        tensor_index = i+1 #starts at 1 becuase we have added a CLS column beforehand
        if df.columns[i] in WORD_COLUMNS:
            token_type_tensor[:, tensor_index] = tensor_dict['WORD_TOKEN']
        elif df.columns[i] in VALS_COLUMNS:
            token_type_tensor[:, tensor_index] = tensor_dict['VALUE_TOKEN']
        elif df.columns[i] == SMILES_COLUMN:
            token_type_tensor[:, tensor_index] = tensor_dict['SMILES_TOKEN']
        else:
            pass

    return token_type_tensor
