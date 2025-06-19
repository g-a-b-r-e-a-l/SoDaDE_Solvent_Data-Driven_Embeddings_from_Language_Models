import argparse
import textwrap

from dataset import load_dataset
from collate import create_collate_fn
from predict_values import predict_values

from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm.auto import tqdm # For a nice progress bar
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import MultiModalImputationTransformer

from config import (VOCAB_SIZE_COLUMNS, TRANSFORMER_HIDDEN_DIM, 
                    MAX_SEQUENCE_LENGTH, TOKEN_TYPE_VOCAB_SIZE, 
                    NUM_ATTENTION_HEADS, NUM_TRANSFORMER_LAYERS, 
                    DROPOUT_RATE, DATA_PATH, VAL_PATH, 
                    COLUMN_DICT, MASKING_PROBABILITY, TOKEN_TYPE_VOCAB)


def main(
    number_of_epochs: int,
    learning_rate: float = 0.001,
    batch_size: int = 16,
    shuffle: bool = True,
    DATA_PATH: str = DATA_PATH,
    VAL_PATH: str = VAL_PATH,
    MASKING_PROBABILITY: float = MASKING_PROBABILITY,
    DROPOUT_RATE: float = DROPOUT_RATE

):
    
    #Load the training and validation datasets
    dataset_train, chemberta_dimension = load_dataset(DATA_PATH, COLUMN_DICT, MAX_SEQUENCE_LENGTH)
    dataset_val, _ = load_dataset(VAL_PATH, COLUMN_DICT, MAX_SEQUENCE_LENGTH)

    #Wrap collate function to take additional variables
    configured_collate_fn = create_collate_fn(TOKEN_TYPE_VOCAB, MASKING_PROBABILITY)


    # Create DataLoader for training and validation datasets
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, collate_fn=configured_collate_fn)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=shuffle, collate_fn=configured_collate_fn)

    # Initialize the model
    model = MultiModalImputationTransformer(
         chemberta_fp_dim=chemberta_dimension,
         column_vocab_size=VOCAB_SIZE_COLUMNS,
         transformer_hidden_dim=TRANSFORMER_HIDDEN_DIM,
         max_sequence_length=MAX_SEQUENCE_LENGTH,
         token_type_vocab_size=TOKEN_TYPE_VOCAB_SIZE,
         num_attention_heads=NUM_ATTENTION_HEADS,
         num_transformer_layers=NUM_TRANSFORMER_LAYERS,
         dropout_rate=DROPOUT_RATE
     )
    
    # Set up the optimizer and loss function and learning rate scheduler

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    criterion = nn.MSELoss()

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',         # 'min' for loss, 'max' for accuracy/F1 score
        factor=0.5,         # Factor by which the learning rate will be reduced. new_lr = lr * factor
        patience=5,         # Number of epochs with no improvement after which learning rate will be reduced.
        #verbose=True,       # Print a message when LR is reduced
        min_lr=1e-7,        # Minimum learning rate to which it can be reduced
        cooldown=0          # Number of epochs to wait before resuming normal operation after lr has been reduced.
    )

    # Training loop
    best_val_loss = float('inf') # Initialize with a very large number
    best_model_path = f'val_loss{best_val_loss}_DP_{DATA_PATH}_LR_{learning_rate}_DPR_{DROPOUT_RATE}_MP_{MASKING_PROBABILITY}.pth'

    for epoch in range(number_of_epochs):
        model.train() # Set the model to training mode

        # Use tqdm for a progress bar
        # 'desc' is the description, 'leave' keeps the bar after completion
        # 'position=0' helps if you have nested progress bars

        train_loss = predict_values(model, dataloader_train, optimizer, criterion, number_of_epochs, train=True, epoch=epoch)
    
        # 7. Validation step    
   
        model.eval()

        with torch.no_grad():
            val_loss = predict_values(model, dataloader_val, optimizer, criterion, number_of_epochs, train=False, epoch=epoch)
        
        #Update the learning rate scheduler
        scheduler.step(val_loss)

    if val_loss < best_val_loss:
        print(f'Validation loss decreased ({best_val_loss:.4f} --> {val_loss:.4f}). Saving model...')
        best_val_loss = val_loss
        # Save the model's state_dict
        torch.save(model.state_dict(), best_model_path)
    # save results here

    return None


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train transformer.",
        epilog=textwrap.dedent(
            """To pass in arbitrary options, use the -c flag.
            Example usage:
                python train_model.py -learning_rate 0.001 -bs 32 -num_epochs 10
            """
        ),
    )
    argparser.add_argument("-num_epochs", "--number_of_epochs", type=int, help="Number of epochs to train the model.")
    argparser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    argparser.add_argument("-bs", "--batch_size", type=int, default=16, help="Batch size for training.")
    argparser.add_argument("-s", "--shuffle", type=bool, help="Shuffle the dataset before training.")
    argparser.add_argument("-td", "--train_data", type=str, help="Path to the training data file.")
    argparser.add_argument("-vd", "--val_data", type=str, help="Path to the validation data file.")
    argparser.add_argument("-mp", "--masking_probability", type=float, default=0.3, help="Probability of masking tokens in the input.")
    argparser.add_argument("-dr", "--dropout_rate", type=float, default=0.3, help="Dropout rate for the model.")

    args = argparser.parse_args()

    results = main(
        number_of_epochs=args.number_of_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
    )
