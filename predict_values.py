
from config import (
    BATCH_SIZE,
    MAX_SEQUENCE_LENGTH
)
import torch
from tqdm import tqdm

def predict_values(model, dataloader, optimizer, criterion, num_epochs, train=True, epoch=0):

    total_loss = 0

    for batch_idx, batch_dict in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)):
        # 1. Move batch data to the appropriate device
        # This loop ensures all tensors in the dictionary are on the correct device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in batch_dict.items()}

        if train:
            # 2. Zero the gradients
            optimizer.zero_grad()

        # 3. Forward pass
        predicted_values = model(BATCH_SIZE, MAX_SEQUENCE_LENGTH, **batch_dict)

    true_masked_labels = inputs['masked_lm_labels'][inputs['masked_lm_labels'] != -100.0]

        # Check if there are any masked values in the current batch to avoid error
    if predicted_values.numel() > 0 and true_masked_labels.numel() > 0:
        loss = criterion(predicted_values, true_masked_labels)
        total_loss += loss.item()

        if train:
            # 5. Backward pass
            loss.backward()

            # 6. Update model parameters
            optimizer.step()

        average_loss = total_loss / (len(dataloader))
        return average_loss

    else:
        # If no values were masked in this batch, or no valid labels, skip loss calculation
        # and print a warning for debugging. This might happen with very small batches
        # or low masking probability.
        # In a real scenario, you might want to adjust masking or batching to ensure
        # masked values are always present.
        print(f"Warning: No masked values for prediction in batch {batch_idx+1} of epoch {epoch+1}. Skipping loss calculation for this batch.")
        loss = torch.tensor(0.0, device=device) # Assign a zero loss to not affect gradients negatively
        pass