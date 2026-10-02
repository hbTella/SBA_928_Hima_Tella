# Enhancing Market Research with AI Prompt Engineering

## Project Overview

This project explores how prompt engineering and fine-tuning can be used to analyze customer reviews for market research.

I used the Amazon Fine Food Reviews dataset and the pretrained `google/flan-t5-small` model.

The project has two main parts:

* Testing different prompt engineering techniques
* Fine-tuning the model for customer sentiment classification

## Dataset

The dataset used in this project is the Amazon Fine Food Reviews dataset.

The original `SampleData.csv` file contains 10,000 customer reviews.

For fine-tuning, I used a smaller sample of 100 reviews. Each review was converted into an instruction, context, and target format.

The target sentiment was created from the review score:

* 4–5: Positive
* 3: Neutral
* 1–2: Negative

## Prompt Engineering

I tested different prompt variations for three market research scenarios:

* Customer Sentiment
* Customer Pain Points
* Product Improvement

For each scenario, I tested basic, detailed, and structured prompts.

The results showed that making a prompt longer or adding more instructions did not always produce a better response. Some responses were incomplete or too general.

The prompt experiment results are saved in:

prompt_variation_results.csv

## Fine-Tuning

For fine-tuning, I created 100 structured examples and divided them into:

* 70 training examples
* 15 validation examples
* 15 held-out evaluation examples

The `google/flan-t5-small` model was fine-tuned for 3 epochs using the CPU.

The training and validation losses were recorded in:

`training_log.csv`

The fine-tuned model was saved in:

`fine_tuned_model/`

## Model Evaluation

I compared the original pretrained model with the fine-tuned model using the same 15 held-out reviews.

The results were:

* Base model accuracy: **80.00%**
* Fine-tuned model accuracy: **86.67%**

The fine-tuned model correctly classified 13 out of 15 evaluation reviews, while the base model correctly classified 12 out of 15.

The complete comparison is available in:

`model_comparison_results.csv`

## Bias and Fairness

The dataset may contain potential bias because the project uses a small sample of customer reviews. There may also be differences in sentiment, product categories, review styles, and customer experiences.

To reduce potential bias, a larger and more balanced dataset could be used. The model should also be tested on different types of reviews and product categories.

## Project Files

* `SampleData.csv` – original review dataset
* `src/sba_928_hima_tella/prompt_variations.py` – prompt engineering experiment
* `prompt_variation_results.csv` – prompt experiment results
* `create_finetuning_data.py` – creates the fine-tuning dataset
* `finetuning_dataset.csv` – structured fine-tuning data
* `split_dataset.py` – creates training, validation, and evaluation sets
* `train.csv` – training data
* `validation.csv` – validation data
* `evaluation.csv` – held-out evaluation data
* `tokenize_data.py` – tokenization and preprocessing
* `fine_tune_model.py` – model fine-tuning
* `training_log.csv` – training and validation loss
* `fine_tuned_model/` – saved fine-tuned model
* `evaluate_models.py` – compares the base and fine-tuned models
* `model_comparison_results.csv` – evaluation results
* `README.md` – project documentation

## Conclusion

This project helped me understand the difference between prompt engineering and fine-tuning. Prompt engineering changes the instructions given to a model, while fine-tuning updates the model using task-specific training examples.

The evaluation showed that the fine-tuned model achieved higher accuracy on this small held-out evaluation set, but more data and testing would be needed to draw stronger conclusions.
