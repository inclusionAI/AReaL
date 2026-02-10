"""Evaluation prompts for judge models."""

EVAL_SYSTEM_PROMPT = (
    "You are an intelligent chatbot designed for evaluating the correctness of generative outputs "
    "for question-answer pairs.\n"
    "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully.\n"
    "------\n"
    "##INSTRUCTIONS:\n"
    "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
    "- Consider synonyms or paraphrases as valid matches.\n"
    "- Evaluate the correctness of the prediction compared to the answer."
)

EVAL_QUERY_PROMPT = (
    "I will give you a question related to an image and the following text as inputs:\n\n"
    "1. **Question Related to the Image**: {question}\n"
    "2. **Ground Truth Answer**: {ground_truth}\n"
    "3. **Model Predicted Answer**: {prediction}\n\n"
    "Your task is to evaluate the model's predicted answer against the ground truth answer, "
    "based on the context provided by the question related to the image. Consider the following criteria for evaluation:\n"
    "- **Relevance**: Does the predicted answer directly address the question posed?\n"
    "- **Accuracy**:\n"
    "(1) If the ground truth answer is open-ended, consider whether the prediction reflects the information "
    "given in the ground truth without introducing factual inaccuracies.\n"
    "(2) If the ground truth answer is a definitive answer, strictly compare the model's prediction to the actual answer. "
    "Pay attention to unit conversions such as length and angle, etc. As long as the results are consistent, the model's "
    "prediction should be deemed correct.\n\n"
    "**Output Format**:\n"
    "Your response should include an integer score indicating the correctness of the prediction: 1 for correct and 0 for incorrect.\n"
    'The format should be "Score: 0 or 1"'
)
