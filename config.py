# --- 0. Configuration ---
DATA_PATH = "7376_train_dataset_norm.csv"

WORD_COLUMNS = ['solvent', '1', '3', '5', '7', '9', '11', '13', '15', '17', '19', '21',  '23']
VALS_COLUMNS =  [ '2', '4', '6', '8', '10', '12', '14', '16', '18', '20', '22', '24']
#VALS_COLUMNs =  [ '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']
WORD_TOKENS = ['alkane', 'aromatic', 'halohydrocarbon', 'ether', 'ketone', 'ester', 'nitrile', 'amine', 'amide', 'misc_N_compound', 'carboxylic_acid', 'monohydric_alcohol' , 'polyhydric_alcohol', 'other','ET30', 'alpha', 'beta', 'pi_star', 'SA', 'SB', 'SP', 'SdP', 'N_mol_cm3', 'n', 'fn', 'delta']

TRANSFORMER_HIDDEN_DIM = 384
NUM_ATTENTION_HEADS = 4
NUM_TRANSFORMER_LAYERS = 2

TOKEN_TYPE_WORD = 0
TOKEN_TYPE_SMILES = 1
TOKEN_TYPE_NUM_VALUE = 2
TOKEN_TYPE_MASK = 3
TOKEN_TYPE_CLS = 4
TOKEN_TYPE_SEP = 5

TOKEN_TYPES = ['WORD_TOKEN', 'SMILES_TOKEN', 'VALUE_TOKEN', 'MASK_TOKEN', 'CLS_TOKEN', 'SEP_TOKEN']

TOKEN_TYPE_VOCAB_SIZE = 6

MAX_SEQUENCE_LENGTH = 28
TRANSFORMER_DROPOUT_RATE = 0.1
MASKING_PROBABILITY = 0.2

SMILES_COLUMN = None
CHEMBERTA_FP_DIM = None
VOCAB_SIZE_COLUMNS = None