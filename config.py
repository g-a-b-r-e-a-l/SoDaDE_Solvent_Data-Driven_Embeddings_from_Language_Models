# --- 0. Model Configuration ---
DATA_PATH = "7376_train_dataset_norm.csv"
VAL_PATH = "560_val_dataset_norm.csv"

COLUMN_DICT = {
    'WORD_COLUMNS': ['solvent', '1', '3', '5', '7', '9', '11', '13', '15', '17', '19', '21',  '23'],
    'VALUE_COLUMNS': ['2', '4', '6', '8', '10', '12', '14', '16', '18', '20', '22', '24'],
    'SMILES_COLUMNS': 'SMILES'
}

TOKEN_TYPES = ['WORD_TOKEN', 'SMILES_TOKEN', 'VALUE_TOKEN', 'MASK_TOKEN', 'CLS_TOKEN', 'SEP_TOKEN']
WORD_TOKENS = ['alkane', 'aromatic', 'halohydrocarbon', 'ether', 'ketone', 'ester', 'nitrile', 'amine', 'amide', 'misc_N_compound', 'carboxylic_acid', 'monohydric_alcohol' , 'polyhydric_alcohol', 'other','ET30', 'alpha', 'beta', 'pi_star', 'SA', 'SB', 'SP', 'SdP', 'N_mol_cm3', 'n', 'fn', 'delta']
VOCAB_SIZE_COLUMNS = len(WORD_TOKENS)
TOKEN_TYPE_VOCAB_SIZE = len(TOKEN_TYPES)
token_type_vocab = {token_type: i for i, token_type in enumerate(TOKEN_TYPES)}
TOKEN_TYPE_VOCAB = token_type_vocab

TRANSFORMER_HIDDEN_DIM = 384
NUM_ATTENTION_HEADS = 4
NUM_TRANSFORMER_LAYERS = 2


MAX_SEQUENCE_LENGTH = 28
DROPOUT_RATE = 0.3
MASKING_PROBABILITY = 0.3

#--- 1. Training Configuration ---

NUM_EPOCHS = 40
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
