import argparse
import textwrap

from dataset import load_dataset
from collate import collate_fn

from torch.utils.data import DataLoader

from models import MultiModalImputationTransformer

from config import VOCAB_SIZE_COLUMNS, TRANSFORMER_HIDDEN_DIM, MAX_SEQUENCE_LENGTH, TOKEN_TYPE_VOCAB_SIZE, NUM_ATTENTION_HEADS, NUM_TRANSFORMER_LAYERS, TRANSFORMER_DROPOUT_RATE


def main(
    number_of_epochs: int,
    learning_rate: float,
    batch_size: int = 32,
    shuffle: bool = True,

):
    # df = load_dataset()
    # chemberta = load_chemberta(tokenizer)

    # dataloader = DataLoader(df, batch_size, shuffle=shuffle, collate_fn=collate_fn)

    # model = MultiModalImputationTransformer(
    #     chemberta_fp_dim=CHEMBERTA_FP_DIM,
    #     column_vocab_size=VOCAB_SIZE_COLUMNS,
    #     transformer_hidden_dim=TRANSFORMER_HIDDEN_DIM,
    #     max_sequence_length=MAX_SEQUENCE_LENGTH,
    #     token_type_vocab_size=TOKEN_TYPE_VOCAB_SIZE,
    #     num_attention_heads=NUM_ATTENTION_HEADS,
    #     num_transformer_layers=NUM_TRANSFORMER_LAYERS,
    #     dropout_rate=TRANSFORMER_DROPOUT_RATE
    # )

    # train code here

    # save results here

    output = f"Would have trained a model with the following parameters:\n" \
                f"Number of epochs: {number_of_epochs}\n" \
                f"Learning rate: {learning_rate}\n" \
                f"Batch size: {batch_size}\n" \
                f"Shuffle: {shuffle}\n"
    
    # save as txt file
    with open("results/training_results.txt", "w") as f:
        f.write(output)

    return None


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train transformer.",
        epilog=textwrap.dedent(
            """To pass in arbitrary options, use the -c flag.
            Example usage:
                python train_model.py -hidden_dim 128 -learning_rate 0.001 -bs 32 -num_epochs 10
            """
        ),
    )
    argparser.add_argument("-num_epochs", "--number_of_epochs", type=int, help="Number of epochs to train the model.")
    argparser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    argparser.add_argument("-bs", "--batch_size", type=int, default=32, help="Batch size for training.")
    argparser.add_argument("-s", "--shuffle", type=bool, help="Shuffle the dataset before training.")

    args = argparser.parse_args()

    results = main(
        args.number_of_epochs,
        args.learning_rate,
        args.batch_size,
        args.shuffle,
    )
