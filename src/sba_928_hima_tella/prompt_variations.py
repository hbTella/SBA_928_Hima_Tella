# Import pandas for reading and working with the CSV dataset.
import pandas as pd

# Import the tokenizer and model classes from Hugging Face Transformers.
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# This function sends a prompt to the pretrained FLAN-T5 model
# and returns the model's generated response as readable text.
def generate_response(model, tokenizer, prompt):

    # Convert the text prompt into numerical tokens that the model can understand.
    # return_tensors="pt" converts the tokens into PyTorch tensors.
    # truncation=True prevents very long prompts from exceeding the model's input limit.
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )

    # Generate a response using the pretrained model.
    # **inputs passes the tokenized prompt information to the model.
    # max_new_tokens=80 limits the response to a maximum of 80 newly generated tokens.
    outputs = model.generate(
        **inputs,
        max_new_tokens=80
    )

    # Convert the generated tokens back into normal human-readable text.
    # outputs[0] selects the response for our single input.
    # skip_special_tokens=True removes special tokens used internally by the model.
    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


def main():
    # Load the dataset
    df = pd.read_csv("SampleData.csv")

    # Select one customer review for the prompt experiment
    review = df.iloc[0]["Text"]

    print("Customer Review:")
    print(review)

    # Load the pretrained FLAN-T5-small model
    model_name = "google/flan-t5-small"

    print("\nLoading pretrained model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Prompt variations
    prompts = {
        "Scenario 1 - Basic Sentiment": (
            f"Determine whether the customer review is positive, neutral, or negative.\n"
            f"Customer Review: {review}"
        ),

        "Scenario 1 - Detailed Sentiment": (
            f"Analyze the customer review and classify the sentiment as positive, "
            f"neutral, or negative. Explain the main reason for the classification.\n"
            f"Customer Review: {review}"
        ),

        "Scenario 1 - Structured Sentiment": (
            f"Analyze the customer review and provide:\n"
            f"- Sentiment\n"
            f"- Main reason for the sentiment\n"
            f"- Key customer concern or positive point\n"
            f"Customer Review: {review}"
        ),

        "Scenario 2 - Basic Pain Point": (
            f"Identify the main problem mentioned in the customer review.\n"
            f"Customer Review: {review}"
        ),

        "Scenario 2 - Detailed Pain Point": (
            f"Analyze the review and identify the customer's main pain point. "
            f"Explain how this issue affected the customer's experience.\n"
            f"Customer Review: {review}"
        ),

        "Scenario 2 - Structured Pain Point": (
            f"Analyze the review and provide:\n"
            f"- Main pain point\n"
            f"- Evidence from the review\n"
            f"- Possible impact on customer satisfaction\n"
            f"Customer Review: {review}"
        ),

        "Scenario 3 - Basic Improvement": (
            f"Suggest an improvement based on the customer review.\n"
            f"Customer Review: {review}"
        ),

        "Scenario 3 - Detailed Improvement": (
            f"Analyze the customer feedback and identify one specific improvement "
            f"that could make the product or customer experience better.\n"
            f"Customer Review: {review}"
        ),

        "Scenario 3 - Market Research": (
            f"Act as a market research analyst. Analyze the customer feedback, "
            f"identify the main issue, and suggest a practical product or "
            f"customer-experience improvement based only on the information provided.\n"
            f"Customer Review: {review}"
        ),
    }

    # Store experiment results
    results = []

    # Run all prompt variations
    for name, prompt in prompts.items():
        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)

        print("\nPrompt:")
        print(prompt)

        response = generate_response(model, tokenizer, prompt)

        print("\nModel Response:")
        print(response)

        # Save the result
        results.append({
            "Prompt Variation": name,
            "Customer Review": review,
            "Prompt": prompt,
            "Model Response": response
        })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("prompt_variation_results.csv", index=False)

    print("\nResults saved to: prompt_variation_results.csv")


if __name__ == "__main__":
    main()