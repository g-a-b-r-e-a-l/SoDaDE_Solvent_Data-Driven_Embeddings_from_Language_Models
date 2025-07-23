import torch

def create_collate_fn(token_type_vocab, masking_probability=0.15):
    """
    Returns a collate_fn that closes over token_type_vocab and masking_probability.
    """
    def collate_fn_with_args(batch_list_of_dicts):
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

        attention_mask = torch.ones(token_type_ids.shape, dtype=torch.bool)
        masked_lm_labels = torch.full((token_type_ids.shape), -100.0, dtype=torch.float)
        final_mask = torch.zeros_like(masked_lm_labels, dtype=torch.bool)

        present_vals = ~torch.isnan(values_ref)
        value_positions = token_type_ids == token_type_vocab['VALUE_TOKEN']
        selected_positions = present_vals & value_positions

        rand_tensor = torch.rand(values_ref.shape)
        final_mask = (selected_positions) & (rand_tensor < masking_probability)

        masked_lm_labels[final_mask]= values_ref[final_mask]
        token_type_ids[final_mask] = token_type_vocab['MASK_TOKEN']

        token_type_ids[missing_val_mask] = token_type_vocab['MASK_TOKEN'] # Apply mask for existing missing values

        return {
            'SMILES_fps' : SMILES_fps,
            'word_tokens_ref' : word_tokens_ref,
            'values_ref' : values_ref,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'masked_lm_labels': masked_lm_labels
        }
    return collate_fn_with_args