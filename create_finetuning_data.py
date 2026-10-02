import pandas as pd


def main():
    # Load the Amazon Fine Food Reviews dataset
    df = pd.read_csv("SampleData.csv")

    # Use a small subset for fine-tuning
    df = df.head(100).copy()

    # Convert the review score into a sentiment label
    def get_sentiment(score):
        if score >= 4:
            return "positive"
        elif score == 3:
            return "neutral"
        else:
            return "negative"

    # Create the target sentiment
    df["target"] = df["Score"].apply(get_sentiment)

    # Create the instruction for the model
    df["instruction"] = (
        "Determine whether the customer review is positive, neutral, or negative."
    )

    # Use the customer review as the context
    df["context"] = df["Text"]

    # Keep only the columns needed for fine-tuning
    training_data = df[
        ["instruction", "context", "target"]
    ]

    # Save the fine-tuning dataset
    training_data.to_csv(
        "finetuning_dataset.csv",
        index=False
    )

    print("Fine-tuning dataset created successfully.")
    print("Dataset shape:", training_data.shape)
    print("\nFirst 5 examples:")
    print(training_data.head())


if __name__ == "__main__":
    main()