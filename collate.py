import torch
from config import MASKING_PROBABILITY, TOKEN_TYPE_MASK

def collate_fn(batch_list_of_dicts):

    values_ref_list = [d['values_tensor'] for d in batch_list_of_dicts]
    missing_val_mask_list = [d['missing_val_mask'] for d in batch_list_of_dicts]
    word_tokens_ref_list = [d['word_index_tensor'] for d in batch_list_of_dicts]
    token_type_ids_list = [d['token_type_tensor'] for d in batch_list_of_dicts]
    SMILES_fps_list = [d['chemberta_fps_tensor'] for d in batch_list_of_dicts]

    values_ref = torch.stack(values_ref_list)
    missing_val_mask = torch.stack(missing_val_mask_list)
    word_tokens_ref = torch.stack(word_tokens_ref_list)
    token_type_ids = torch.stack(token_type_ids_list)
    SMILES_fps = torch.stack(SMILES_fps_list)

    sequence_ids = torch.arange(token_type_ids.shape[1])
    position_ids = sequence_ids.unsqueeze(0).expand(token_type_ids.shape[0], -1)

    attention_mask = torch.ones(token_type_ids.shape, dtype=torch.bool)
    masked_lm_labels = torch.full((token_type_ids.shape), -100.0, dtype=torch.float)
    final_mask = torch.zeros_like(masked_lm_labels, dtype=torch.bool)

    present_vals = ~torch.isnan(values_ref)
    rand_tensor = torch.rand(values_ref.shape)
    final_mask = (present_vals) & (rand_tensor < MASKING_PROBABILITY)

    masked_lm_labels[final_mask]= values_ref[final_mask]
    token_type_ids[final_mask] = TOKEN_TYPE_MASK
    token_type_ids[missing_val_mask] = TOKEN_TYPE_MASK

    return {
        'SMILES_fps' : SMILES_fps,
        'word_tokens_ref' : word_tokens_ref,
        'values_ref' : values_ref,
        'token_type_ids': token_type_ids,
        'position_ids': position_ids,
        'attention_mask': attention_mask,
        'masked_lm_labels': masked_lm_labels
    }