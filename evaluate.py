import json
import os
import time
from tqdm import tqdm
import dspy

from argparse import ArgumentParser
from key import OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, DEEP_SEEK_API_KEY
from validate import validate_wrong_assumptions  # Ensure this function is implemented in validate.py

# ===============================
# Configuration
# ===============================
TEMPERATURE = 0.7  # Temperature for LLM generation
SLEEP_TIME = 0  # Optional delay to avoid rate limits
MAX_OUTPUT = 10  # Limit number of processed questions; set to None to process all

# Model lists - Uncomment to activate specific models
OPENAI_MODELS = ['openai/gpt-4-turbo', 'openai/gpt-3.5-turbo', 'openai/gpt-4o']  # e.g., ['openai/gpt-4-turbo']
ANTHROPIC_MODELS = ['anthropic/claude-3-5-sonnet-20240620']  # e.g., ['anthropic/claude-3-5-sonnet-20240620', 
GOOGLE_MODELS = ['gemini/gemini-1.5-pro']  # e.g., ['gemini/gemini-1.5-pro']
DEEP_SEEK_MODELS = ['deepseek/deepseek-chat',
                    'deepseek/deepseek-reasoner']  # e.g., ['deepseek/deepseek-chat', 'deepseek/deepseek-reasoner']

# Input and Output paths
INPUT_JSON_PATH = "data/all_data.json"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================
# Helper Functions
# ===============================

def load_questions(json_file_path, max_output=None):
    """Load questions and assumptions from JSON file, with optional size limit."""
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    questions = [
        {
            'example_question': item.get('example_question', ''),
            'example_assumption': item.get('example_assumption', ''),
            'from_model': item.get('from_model', {})  # Preserve original model info if available
        }
        for item in data
    ]

    return questions if max_output is None else questions[:max_output]


def process_questions(questions, model_name, api_key, output_json_path):
    """Generate answers using specified model, validate them, and write results to file."""
    # Set up LLM for answer generation
    lm = dspy.LM(model_name, api_key=api_key, temperature=TEMPERATURE)
    dspy.configure(lm=lm)

    # Use a consistent validator (e.g., GPT-4o)
    validator = dspy.LM('openai/gpt-4o', api_key=OPENAI_API_KEY, temperature=TEMPERATURE)

    results = []

    for item in tqdm(questions, desc=f"Processing with {model_name}", unit="question"):
        time.sleep(SLEEP_TIME)  # Prevent hitting API rate limits

        question = item['example_question']
        assumption = item['example_assumption']
        from_model = item.get('from_model', {})

        try:
            # Generate answer
            answer = lm(question)[0]

            # Format for validation
            validation_data = [{
                'Question': question,
                'Answer': answer,
                'Wrong Assumption': assumption
            }]

            # Validate the answer against the assumption
            validation_results, scores = validate_wrong_assumptions(validator, data=validation_data)

            # Append result
            results.append({
                'model': model_name,
                'question': question,
                'information': assumption,
                'model_answer': answer,
                'evaluation': scores,
                'from_model': from_model
            })

            # Write to output file continuously to avoid data loss
            with open(output_json_path, 'w', encoding='utf-8') as file:
                json.dump(results, file, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"Error processing question: {question[:50]}... | Error: {e}")


# ===============================
# Main Evaluation Runner
# ===============================

def run_all_models():
    """Evaluate all configured models on the dataset."""
    questions = load_questions(INPUT_JSON_PATH, MAX_OUTPUT)

    model_configs = [
        (OPENAI_MODELS, OPENAI_API_KEY),
        (ANTHROPIC_MODELS, ANTHROPIC_API_KEY),
        (GOOGLE_MODELS, GOOGLE_API_KEY),
        (DEEP_SEEK_MODELS, DEEP_SEEK_API_KEY)
    ]

    for model_list, api_key in model_configs:
        for model_name in model_list:
            print(f"\nEvaluating model: {model_name}")
            output_filename = f"evaluation_results_{model_name.replace('/', '_')}.json"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            process_questions(questions, model_name, api_key, output_path)


def run_single_model(args):
    """Evaluate a single model on the dataset."""
    questions = load_questions(INPUT_JSON_PATH, MAX_OUTPUT)
    print(f"\nEvaluating model: {args.model_name}")
    output_filename = f"evaluation_results_{args.model_name.replace('/', '_')}.json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    process_questions(questions, args.model_name, args.api_key, output_path)


# ===============================
# Entry Point
# ===============================

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default="single", choices=["reproduce", "single"])
    parser.add_argument("--model_name", type=str, default="openai/gpt-4o")
    parser.add_argument("--api_key", type=str, default=OPENAI_API_KEY)

    args = parser.parse_args()
    if args.mode == "single":
        run_single_model(args)
    elif args.mode == "reproduce":
        run_all_models()
    else:
        print("Invalid mode selected. Use 'reproduce' or 'single'.")
