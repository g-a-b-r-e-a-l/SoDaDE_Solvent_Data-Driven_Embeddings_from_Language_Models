import pandas as pd

from torch.utils.data import Dataset
from utils import create_diff_token_vocabs, create_values_tensor, word_token_indicies, create_token_type_tensor, create_fingerprints
from utils import load_and_clean_csv

from config import WORD_TOKENS, TOKEN_TYPES, DATA_PATH, MAX_SEQUENCE_LENGTH, WORD_COLUMNS, VALS_COLUMNS

class MolecularPropertyDataset(Dataset):
    def __init__(self, df, max_len_plus_2, word_columns, vals_columns, smiles_column,
                 tokenizer, model_chemberta):
        self.df = df
        self.token_type_vocab, self.word_vocab, VOCAB_SIZE_COLUMNS = create_diff_token_vocabs(WORD_TOKENS, TOKEN_TYPES)

        self.values_tensor, self.missing_val_mask = create_values_tensor(df, max_len_plus_2, vals_columns)
        self.word_index_tensor = word_token_indicies(df, max_len_plus_2, word_columns, self.word_vocab)
        self.token_type_tensor = create_token_type_tensor(df, max_len_plus_2, word_columns, vals_columns, smiles_column, self.token_type_vocab)

        self.chemberta_fps_tensor = create_fingerprints(df, max_len_plus_2, smiles_column, tokenizer, model_chemberta)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        batch_dict = {
            'values_tensor': self.values_tensor[idx],
            'missing_val_mask': self.missing_val_mask[idx],
            'word_index_tensor': self.word_index_tensor[idx],
            'token_type_tensor': self.token_type_tensor[idx],
            'chemberta_fps_tensor': self.chemberta_fps_tensor[idx]
        }

        return batch_dict


def load_dataset(tokenizer, model_chemberta):
    df = load_and_clean_csv(DATA_PATH)

    if df is None:
        raise ValueError("DataFrame is empty or not loaded correctly.")

    tokenizer = None  # Replace with actual tokenizer initialization
    model_chemberta = None  # Replace with actual model initialization

    dataset = MolecularPropertyDataset(df, 
                                       MAX_SEQUENCE_LENGTH, 
                                       WORD_COLUMNS, 
                                       VALS_COLUMNS, 
                                       'SMILES', 
                                       tokenizer, 
                                       model_chemberta)
    
    return dataset