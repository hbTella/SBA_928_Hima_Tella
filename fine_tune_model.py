# Import pandas for reading CSV files and saving the training log.
import pandas as pd

# Import PyTorch for creating the dataset, data loader, and training loop.
import torch
from torch.utils.data import Dataset, DataLoader

# Import the tokenizer and sequence-to-sequence model from Hugging Face.
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# This class prepares our CSV data so it can be used by the model.
class ReviewDataset(Dataset):

    def __init__(self, file_name, tokenizer):

        # Load the dataset from the CSV file.
        df = pd.read_csv(file_name)

        # Combine the instruction and customer review.
        # This becomes the input that the model receives.
        inputs = (
            df["instruction"]
            + "\nCustomer Review: "
            + df["context"]
        )

        # Convert the input text into tokens.
        self.inputs = tokenizer(
            inputs.tolist(),
            max_length=256,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # Convert the expected target responses into tokens.
        self.labels = tokenizer(
            text_target=df["target"].tolist(),
            max_length=16,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )["input_ids"]

        # Ignore padding tokens when calculating the training loss.
        # -100 tells PyTorch not to include those tokens in the loss.
        self.labels[self.labels == tokenizer.pad_token_id] = -100

    def __len__(self):

        # Return the number of examples in the dataset.
        return len(self.labels)

    def __getitem__(self, index):

        # Return one training example.
        return {
            "input_ids": self.inputs["input_ids"][index],
            "attention_mask": self.inputs["attention_mask"][index],
            "labels": self.labels[index]
        }


def main():

    # Name of the pretrained model we are fine-tuning.
    model_name = "google/flan-t5-small"

    print("Loading tokenizer and model...")

    # Load the tokenizer used to convert text into tokens.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the pretrained FLAN-T5-small model.
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Prepare the training and validation datasets.
    train_dataset = ReviewDataset(
        "train.csv",
        tokenizer
    )

    validation_dataset = ReviewDataset(
        "validation.csv",
        tokenizer
    )

    # DataLoader groups examples into batches during training.
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True
    )

    # Validation data does not need to be shuffled.
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=4,
        shuffle=False
    )

    # Use the CPU because this project is running without a GPU.
    device = torch.device("cpu")

    # Move the model to the selected device.
    model.to(device)

    # AdamW is the optimizer used to update the model's parameters.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5
    )

    # Number of times the model will go through the training data.
    epochs = 3

    print("\nStarting fine-tuning...")
    print("Training examples:", len(train_dataset))
    print("Validation examples:", len(validation_dataset))
    print("Epochs:", epochs)

    # Store the training and validation loss for each epoch.
    training_log = []

    # Start the training process.
    for epoch in range(epochs):

        # Put the model into training mode.
        model.train()

        # Keep track of the total training loss.
        total_train_loss = 0

        # Process the training data batch by batch.
        for batch in train_loader:

            # Move the input data to the selected device.
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Clear the gradients from the previous step.
            optimizer.zero_grad()

            # Run the model using the training inputs and targets.
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Get the loss calculated by the model.
            loss = outputs.loss

            # Calculate gradients used to update the model.
            loss.backward()

            # Update the model parameters.
            optimizer.step()

            # Add this batch's loss to the total.
            total_train_loss += loss.item()

        # Calculate the average training loss for the epoch.
        average_train_loss = (
            total_train_loss / len(train_loader)
        )

        # Put the model into evaluation mode.
        model.eval()

        # Keep track of validation loss.
        total_validation_loss = 0

        # Do not calculate gradients during validation.
        with torch.no_grad():

            # Process the validation data batch by batch.
            for batch in validation_loader:

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Calculate the validation loss.
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_validation_loss += outputs.loss.item()

        # Calculate the average validation loss.
        average_validation_loss = (
            total_validation_loss / len(validation_loader)
        )

        # Display the losses for this epoch.
        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Training Loss: {average_train_loss:.4f} - "
            f"Validation Loss: {average_validation_loss:.4f}"
        )

        # Save the loss values so we can review them later.
        training_log.append({
            "epoch": epoch + 1,
            "training_loss": average_train_loss,
            "validation_loss": average_validation_loss
        })

    # Save the fine-tuned model and tokenizer.
    model.save_pretrained("fine_tuned_model")
    tokenizer.save_pretrained("fine_tuned_model")

    # Convert the training log into a DataFrame.
    log_df = pd.DataFrame(training_log)

    # Save the training and validation losses to a CSV file.
    log_df.to_csv(
        "training_log.csv",
        index=False
    )

    print("\nFine-tuning completed.")
    print("Model saved to: fine_tuned_model")
    print("Training log saved to: training_log.csv")


# Run the main function when this file is executed.
if __name__ == "__main__":
    main()