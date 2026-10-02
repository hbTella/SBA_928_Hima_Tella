# Import pandas for reading the evaluation dataset and saving results.
import pandas as pd

# Import the tokenizer and sequence-to-sequence model.
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Generate a response from a model for a given prompt.
def generate_response(model, tokenizer, prompt):

    # Convert the prompt into tokens that the model can understand.
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    # Generate the model's response.
    outputs = model.generate(
        **inputs,
        max_new_tokens=16
    )

    # Convert the generated tokens back into readable text.
    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


def main():

    # Load the held-out evaluation dataset.
    df = pd.read_csv("evaluation.csv")

    # Load the original pretrained FLAN-T5-small model.
    base_model_name = "google/flan-t5-small"

    print("Loading pretrained model...")

    base_tokenizer = AutoTokenizer.from_pretrained(
        base_model_name
    )

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name
    )

    # Load the fine-tuned model that we trained earlier.
    print("Loading fine-tuned model...")

    fine_tuned_tokenizer = AutoTokenizer.from_pretrained(
        "fine_tuned_model"
    )

    fine_tuned_model = AutoModelForSeq2SeqLM.from_pretrained(
        "fine_tuned_model"
    )

    # Store the comparison results.
    results = []

    # Evaluate both models on exactly the same examples.
    for _, row in df.iterrows():

        # Create the same input format used during training.
        prompt = (
            row["instruction"]
            + "\nCustomer Review: "
            + row["context"]
        )

        # Generate response from the original model.
        base_response = generate_response(
            base_model,
            base_tokenizer,
            prompt
        )

        # Generate response from the fine-tuned model.
        fine_tuned_response = generate_response(
            fine_tuned_model,
            fine_tuned_tokenizer,
            prompt
        )

        # Save both responses along with the expected target.
        results.append({
            "Instruction": row["instruction"],
            "Customer Review": row["context"],
            "Expected Target": row["target"],
            "Base Model Response": base_response,
            "Fine-Tuned Model Response": fine_tuned_response
        })

    # Convert the results into a DataFrame.
    results_df = pd.DataFrame(results)

        # Calculate whether each model predicted the expected target.
    results_df["Base Correct"] = (
        results_df["Base Model Response"].str.lower().str.strip()
        == results_df["Expected Target"].str.lower().str.strip()
    )

    results_df["Fine-Tuned Correct"] = (
        results_df["Fine-Tuned Model Response"].str.lower().str.strip()
        == results_df["Expected Target"].str.lower().str.strip()
    )

    # Calculate accuracy for each model.
    base_accuracy = results_df["Base Correct"].mean()
    fine_tuned_accuracy = results_df["Fine-Tuned Correct"].mean()

    print(f"\nBase Model Accuracy: {base_accuracy:.2%}")
    print(f"Fine-Tuned Model Accuracy: {fine_tuned_accuracy:.2%}")

    # Save the side-by-side comparison.
    results_df.to_csv(
        "model_comparison_results.csv",
        index=False
    )

    # Display the comparison in the terminal.
    print("\nModel comparison completed.")

    print("\nResults:")
    print(results_df.to_string(index=False))

    print(
        "\nResults saved to: "
        "model_comparison_results.csv"
    )


if __name__ == "__main__":
    main()