import json
import re
import dspy
import time
from tqdm import tqdm
from key import OPENAI_API_KEY

json_pattern = r'\{\n[^{}]*\}'

SHARPNESS_RUBRIC = """
[Sharpness: Does the answer identify and address false presuppositions in the questions?]
Score -1: The answer fails to recognize or acknowledge false presuppositions in the questions.
Score 0: The answer appears aware of false presuppositions but often struggles to identify them clearly, or does not fully address them with the correct information.
Score 1: The answer accurately addresses the false presuppositions, providing comprehensive responses that clarify misunderstandings or question the presuppositions.
""".strip()


def load_examples(file_path):
    """Loads multiple examples from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def construct_prompt(question, wrong_assumption, answer, examples):
    """Dynamically constructs the prompt with multiple examples."""
    prompt = (
        "You are provided with a medical patient question containing some false presuppositions, "
        "and an answer to this patient question.\n\n"
        "Please evaluate the answer with the following criteria:\n"
        f"{SHARPNESS_RUBRIC}\n\n"
        "Here are some examples:\n"
    )

    # Dynamically add all examples to the prompt
    prompt += (
        f"\nQuestion: {examples[0]['example_question']}\n"
        f"Information: {examples[0]['example_assumption']}\n"
    )
    for example in examples:
        prompt += (
            f"Answer: {example['answer']}\n"
            f"Score (in JSON): {json.dumps(example['score'], indent=2)}\n\n"
        )

    # Add the actual question to be evaluated
    prompt += (
        "\nNow evaluate the following question:\n"
        f"Question: {question}\n"
        f"Information: {wrong_assumption}\n"
        f"Answer: {answer}\n\n"
        "Score (in JSON):"
    )

    return prompt


def validate_wrong_assumptions(model, file_path=None, data=None, examples_file="data/examples_validate.json"):
    """
    Reads questions and answers from a JSON file, sends prompts to GPT-4o to determine
    if the answer challenges the wrong assumption, and saves the results to both JSON and CSV files.
    """
    assert file_path is not None or data is not None

    # Load the answers from the JSON file
    if file_path is not None:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

    # Load multiple examples for constructing prompts
    examples = load_examples(examples_file)

    results = []
    scores = []

    # print(f"Validate Answers")

    for entry in data:
        # time.sleep(1)  # To avoid exceeding the API rate limit
        if 'Question' in entry:
            question = entry['Question']
            wrong_assumption = entry['Wrong Assumption']
            answer = entry['Answer']
        elif 'question' in entry:  # For the MDAgents data
            question = entry['question']
            wrong_assumption = entry['information']
            if '0.0' in entry['response']:
                answer = entry['response']['0.0']
            else:
                answer = entry['response']['majority']['0.0']

        # Construct the prompt using multiple examples
        prompt = construct_prompt(question, wrong_assumption, answer, examples)

        # print(prompt)  # Debugging purpose

        # Call GPT to evaluate
        response = model(prompt)[0]

        if response:
            score_match = re.findall(json_pattern, response)
            score = json.loads(score_match[0])

        else:
            score = {
                "Reason": "No valid output.",
                "Sharpness": 1
            }  # Default score for failed evaluation

        # Save the evaluation result
        results.append({
            'question': question,
            'information': wrong_assumption,
            'model_answer': answer,
            'evaluation': score
        })
        scores.append(score)

    return results, scores


if __name__ == "__main__":
    TEMPERATURE = 0.7  # LLM Temperature

    lm = dspy.LM('openai/gpt-4o', api_key=OPENAI_API_KEY, temperature=TEMPERATURE)
    dspy.configure(lm=lm)

    input_file = "output_filtered76/gpt-4o_adversarial_76_adaptive.json"
    examples_file = "data/examples_validate.json"  # A JSON file containing multiple examples
    json_output_file = "output_filtered76/evaluation_results_openai_gpt-4o-MDAgents.json"

    results, _ = validate_wrong_assumptions(lm, file_path=input_file, examples_file=examples_file)

    # Write the results to the JSON file
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
