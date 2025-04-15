import json
import dspy
from key import OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY

from argparse import ArgumentParser
from gen_question import generate_similar_questions, cancer_types, load_examples_from_json
from gen_answer import generate_answers
from validate import validate_wrong_assumptions

model_key_map = {
    "openai/gpt-4o": OPENAI_API_KEY,
    "gemini/gemini-1.5-pro": GOOGLE_API_KEY,
    "anthropic/claude-3-5-sonnet-20240620": ANTHROPIC_API_KEY
}


def format_examples_with_myth(examples, myths, new_example_file):
    for example in examples:
        if "source_type" not in example and "source_row" in example:
            example["source_type"] = "myth"
            if example["source_row"] < 0:
                example["source_info"] = {
                    "cancer": "From MD fellows.",
                    "myth": "From MD Fellows.",
                    "fact": "From MD Fellows.",
                    "source": "https://cancercare.org/"
                }
            else:
                example["source_info"] = myths[example["source_row"] - 1]
    with open(new_example_file, "w") as f:
        json.dump(examples, f, indent=2)
    return examples[-1]["source_row"]


def generate_adversarial(args, myths):
    # Load positive and negative example data from the specified JSON files
    examples = load_examples_from_json(args.pos_input_file)
    neg_examples = load_examples_from_json(args.neg_input_file)
    sampling_examples = [example for example in examples if
                         "score" not in example or example["score"]["Sharpness"] == -1]

    # Construct a list of formatted myth strings; myths holds the raw myth data
    myth_str = ["%s on myth:%s And the fact is %s." % (myth["cancer"].lower(), myth["myth"], myth["fact"]) for myth in
                myths]
    myth_idx = format_examples_with_myth(examples, myths, args.pos_output_file)

    # Set up models
    print("Setting up models...")
    print("Generator", args.generator, "Responser", args.responser, "Validator", args.validator)
    print("Output files", args.pos_output_file, args.neg_output_file, args.log_file)
    generator = dspy.LM(args.generator, api_key=model_key_map[args.generator], temperature=args.temperature,
                        num_retries=20)
    responser = dspy.LM(args.responser, api_key=model_key_map[args.responser], temperature=args.temperature,
                        num_retries=20)
    validator = dspy.LM(args.validator, api_key=model_key_map[args.validator], temperature=args.temperature,
                        num_retries=20)
    dspy.configure(lm=validator)

    # If the generation type is "only-random", skip myth processing
    if args.generate_type == "only-random":
        myth_idx = len(myth_str)
    random_idx = 0
    count = 0  # Track number of positive examples generated
    log = []

    while count < args.d_size:
        # Decide whether to use myth data or random cancer type based on current index
        if myth_idx < len(myth_str):
            # Use myth data: fetch corresponding myth and store source info
            current_myth = myths[myth_idx]
            question_info = myth_str[myth_idx]
            source_type = "myth"
            source_info = current_myth  # Raw myth data
            myth_idx += 1
        elif args.generate_type == "only-myth":
            break
        else:
            # When all myths are used, switch to random cancer type data
            question_info = cancer_types[random_idx]
            source_type = "random"
            source_info = question_info  # Assumes cancer_types elements contain the necessary info
            random_idx = (random_idx + 1) % len(cancer_types)

        # Generate similar questions based on question_info
        retries = 0
        while retries < 3:
            try:
                responses = generate_similar_questions(sampling_examples, neg_examples, question_info, generator)
                break
            except:
                responses = None
                retries += 1
                print("Failed to generate questions.")
        if responses is None:
            continue
        else:
            try:
                answers = generate_answers(responses["qs"], responser)
                results, scores = validate_wrong_assumptions(validator, data=answers)
            except:
                print("Failed to validate answers.")
                continue
            log.append(results)
            for result in results:
                # If Sharpness score is below 0, it's considered a positive example
                if result["evaluation"]["Sharpness"] <= 0:
                    count += 1
                    new_example = {
                        "example_question": result["question"],
                        "example_assumption": result["information"],
                        "answer": result["model_answer"],
                        "score": result["evaluation"],
                        "from_model": args.model_name,
                        "source_type": source_type,  # Indicates whether the source was myth or random
                        "source_info": source_info,  # Corresponding myth or cancer type info
                        "source_row": myth_idx
                    }
                    examples.append(new_example)
                    if result["evaluation"]["Sharpness"] == -1:
                        sampling_examples.append(new_example)
                        print("Pos, Sampling")
                    else:
                        print("Pos")
                    # Append only the new positive examples to pos_output_file
                    with open(args.pos_output_file, "w") as f:
                        json.dump(examples, f, indent=2)
                else:
                    new_neg_example = {
                        "example_question": result["question"],
                        "example_assumption": result["information"],
                        "reason": "This is too easy for LLM to detect: " + result["information"],
                        "answer": result["model_answer"],
                        "score": result["evaluation"],
                        "from_model": args.model_name,
                        "source_type": source_type,
                        "source_info": source_info,
                        "source_row": myth_idx,
                    }
                    neg_examples.append(new_neg_example)
                    print("Neg")
                    # Append only the new negative examples to neg_output_file
                    with open(args.neg_output_file, "w") as f:
                        json.dump(neg_examples, f, indent=2)

        print("Source type:", source_type)
        print("Idx:", myth_idx)
        print("Count Pos Sampling:", len(sampling_examples))
        print("Count Pos All:", len(examples))
        print("Count Neg:", len(neg_examples))

    # Write generation logs to the log file
    json.dump(log, open(args.log_file, "w"), indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--validator", type=str, default="openai/gpt-4o")
    parser.add_argument("--generator", type=str, default="openai/gpt-4o")
    parser.add_argument("--responser", type=str, default="openai/gpt-4o")
    parser.add_argument("--model_tag", type=str, default="gpt-4o",
                        help="the 'from_model' field in the output json file")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--generate_type", type=str, choices=["only-myth", "only-random", "both"],
                        default="only-myth")
    parser.add_argument("--pos_input_file", type=str, default="data/init_examples.json")
    parser.add_argument("--neg_input_file", type=str, default="data/init_neg_examples.json")
    parser.add_argument("--log_file", type=str, default="data/log.json")
    parser.add_argument("--pos_output_file", type=str, required=True)
    parser.add_argument("--neg_output_file", type=str, required=True)
    parser.add_argument("--d_size", type=int, help="Maximum size of generated data", default=5000)
    args = parser.parse_args()

    # Specify path to the myth file
    myth_treament_file = "data/myth_treatment.jsonl"
    myths = []
    with open(myth_treament_file, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                myths.append(obj)
            except json.decoder.JSONDecodeError as e:
                print(f"Error parsing line {line_number}: {e}. Line content: {repr(line)}")

    # Print number of successfully parsed JSON objects
    print(f"Parsed {len(myths)} JSON objects.")

    generate_adversarial(args, myths)
