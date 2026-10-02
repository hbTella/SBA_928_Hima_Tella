import pandas as pd
from transformers import AutoTokenizer


def tokenize_dataset(file_name, tokenizer):
    # Load the dataset
    df = pd.read_csv(file_name)

    # Combine the instruction and customer review
    inputs = (
        df["instruction"]
        + "\nCustomer Review: "
        + df["context"]
    )

    # Tokenize the model inputs
    model_inputs = tokenizer(
        inputs.tolist(),
        max_length=256,
        truncation=True,
        padding="max_length"
    )

    # Tokenize the expected target responses
    labels = tokenizer(
        text_target=df["target"].tolist(),
        max_length=16,
        truncation=True,
        padding="max_length"
    )

    # Add the target token IDs to the tokenized dataset
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def main():
    # Load the FLAN-T5-small tokenizer
    model_name = "google/flan-t5-small"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize each dataset
    train_tokens = tokenize_dataset("train.csv", tokenizer)
    validation_tokens = tokenize_dataset("validation.csv", tokenizer)

    # Print information about the processed datasets
    print("\nTokenization completed.")

    print("Training examples:", len(train_tokens["input_ids"]))
    print("Validation examples:", len(validation_tokens["input_ids"]))

    print("\nNumber of tokens in first training input:")
    print(len(train_tokens["input_ids"][0]))

    print("\nNumber of tokens in first training target:")
    print(len(train_tokens["labels"][0]))


if __name__ == "__main__":
    main()