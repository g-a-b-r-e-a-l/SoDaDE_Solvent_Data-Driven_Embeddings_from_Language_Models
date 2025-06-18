import torch
import torch.nn as nn

from config import TRANSFORMER_HIDDEN_DIM, MAX_SEQUENCE_LENGTH, TOKEN_TYPE_VOCAB_SIZE, VOCAB_SIZE_COLUMNS

class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, chemberta_fp_dim: int, transformer_hidden_dim: int):
        super(FeedForwardNeuralNetwork, self).__init__()

        ff_dimension = 4 * chemberta_fp_dim

        # First linear layer: 10 inputs, 20 outputs
        self.fc1 = nn.Linear(chemberta_fp_dim, ff_dimension)
        # Activation function
        self.relu = nn.ReLU()
        # Second linear layer: 4 * chemberta dimension inputs, 5 outputs
        self.fc2 = nn.Linear(ff_dimension, transformer_hidden_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



class MultiModalInputEmbeddings(nn.Module):
    def __init__(self, chemberta_fp_dim: int, column_vocab_size: int,
                 transformer_hidden_dim: int, max_sequence_length: int,
                 token_type_vocab_size: int, dropout_rate: float):
        super().__init__()

        self.transformer_hidden_dim = transformer_hidden_dim
        #Project the standard sizes of input modalities to d_model
        self.smiles_proj = FeedForwardNeuralNetwork(chemberta_fp_dim, transformer_hidden_dim)
        self.property_embedding = nn.Embedding(column_vocab_size, transformer_hidden_dim)

        # Numerical values are blown-up from a value of 1 to the dimension - retains numerical property
        self.value_proj = nn.Linear(1, transformer_hidden_dim)

        # position encodings and token type embeddings so model understands category of each item in seq
        self.position_embeddings = nn.Embedding(max_sequence_length, transformer_hidden_dim)
        self.token_type_embeddings = nn.Embedding(token_type_vocab_size, transformer_hidden_dim)

        self.LayerNorm = nn.LayerNorm(transformer_hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, batch_size, max_batch_seq_len,
         token_type_vocab: dict,
         SMILES_fps: torch.Tensor, # conains a 2d list of smiles strings, 1 per sequence
         word_tokens_ref: torch.Tensor, # contains the word token index in its position of the sequence
         values_ref: torch.Tensor, # contains the value in its position of the sequence
         token_type_ids: torch.Tensor, # contains token tpyes in order 0 - word, 1 - smiles, 2 - value, 3 - mask, 4 - SEP, 5 - CLS
         position_ids: torch.Tensor, # position of eachitem in seq
         ):

        input_embeddings = torch.zeros(batch_size, max_batch_seq_len, self.transformer_hidden_dim, dtype=torch.float)

        # Create masks for each token type
        word_mask = (token_type_ids == token_type_vocab['WORD_TOKEN'])
        smiles_mask = (token_type_ids == token_type_vocab['SMILES_TOKEN'])
        value_mask = (token_type_ids == token_type_vocab['VALUE_TOKEN'])
        special_token_mask = ((token_type_ids == token_type_vocab['CLS_TOKEN']) & 
                              (token_type_ids == token_type_vocab['SEP_TOKEN']) & 
                              (token_type_ids == token_type_vocab['MASK_TOKEN']))

        # Apply embeddings/projections based on masks
        input_embeddings[word_mask] = self.property_embedding(word_tokens_ref[word_mask])
        input_embeddings[smiles_mask] = self.smiles_proj(SMILES_fps[smiles_mask])
        input_embeddings[value_mask] = self.value_proj(values_ref[value_mask].unsqueeze(-1))
        input_embeddings[special_token_mask] = self.token_type_embeddings(token_type_ids[special_token_mask])

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # Combine all embeddings
        embeddings = input_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class MultiModalImputationTransformer(nn.Module):
    def __init__(self, chemberta_fp_dim: int, column_vocab_size: int,
                 transformer_hidden_dim: int, max_sequence_length: int,
                 token_type_vocab_size: int, num_attention_heads: int,
                 num_transformer_layers: int, dropout_rate: float):
        super().__init__()

        self.hidden_dim = transformer_hidden_dim

        # get embeddings from multimodal embeddings - need to look into this more

        self.embeddings = MultiModalInputEmbeddings(
            chemberta_fp_dim=384,
            column_vocab_size=VOCAB_SIZE_COLUMNS,
            transformer_hidden_dim=TRANSFORMER_HIDDEN_DIM,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            token_type_vocab_size=TOKEN_TYPE_VOCAB_SIZE
        )

        # initialise standard encoder layer - structure - multi-head attention, add+norm, FF, add+norm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=transformer_hidden_dim * 4,
            dropout=dropout_rate,
            batch_first=True
        )

        # stacks encoder layers on top of each other - more means can look at more complex relations
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # regression head for numerical values - GELU might be something to play with
        self.regression_head = nn.Sequential(
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(transformer_hidden_dim),
            nn.Linear(transformer_hidden_dim, 1)
        )

    def forward(self, batch_size, MAX_SEQUENCE_LENGTH,
        SMILES_fps : torch.Tensor,
        word_tokens_ref : torch.Tensor,
        values_ref : torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        masked_lm_labels: torch.Tensor = None
               ):



        embeddings = self.embeddings(batch_size, MAX_SEQUENCE_LENGTH,
         SMILES_fps, # conains a 2d list of smiles strings, 1 per sequence
         word_tokens_ref, # contains the word token index in its position of the sequence
         values_ref, # contains the value in its position of the sequence
         token_type_ids, # contains token tpyes in order 0 - word, 1 - smiles, 2 - value, 3 - mask, 4 - SEP, 5 - CLS
         position_ids, # position of eachitem in seq
         )

        transformer_output = self.transformer_encoder(src=embeddings, src_key_padding_mask=~attention_mask)

        #mask ensures no attendence to masked values. Then
        if masked_lm_labels is not None:
            masked_lm_positions = (masked_lm_labels != -100).nonzero(as_tuple=True)
            masked_token_outputs = transformer_output[masked_lm_positions[0], masked_lm_positions[1]]
            predicted_values = self.regression_head(masked_token_outputs).squeeze(-1)
            return predicted_values
        else:
            return transformer_output