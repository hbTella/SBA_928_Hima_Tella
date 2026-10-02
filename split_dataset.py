import pandas as pd


def main():
    # Load the fine-tuning dataset
    df = pd.read_csv("finetuning_dataset.csv")

    # Shuffle the examples so that the splits are not based on the
    # original order of the reviews.
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Create three separate datasets
    train_df = df[:70]
    validation_df = df[70:85]
    evaluation_df = df[85:100]

    # Save each dataset
    train_df.to_csv("train.csv", index=False)
    validation_df.to_csv("validation.csv", index=False)
    evaluation_df.to_csv("evaluation.csv", index=False)

    print("Dataset split completed.")
    print("Training set:", train_df.shape)
    print("Validation set:", validation_df.shape)
    print("Evaluation set:", evaluation_df.shape)


if __name__ == "__main__":
    main()