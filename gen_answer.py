import json
import time
import dspy
from key import OPENAI_API_KEY

json_pattern = r'\{\n[^{}]*\}'


# Call GPT for each question and generate answers
def generate_answers(parsed_questions, model):
    answered_data = []

    print(f"Generate Answers")

    for idx, item in enumerate(parsed_questions):
        question = item["Question"]
        # cancer_type = item["cancer_type"]
        prompt = question

        # Call GPT model
        try:
            response = model(prompt)[0]
            if response:
                item["Answer"] = response
                answered_data.append(item)
                # print(f"Answer: {reply_content}\n")
            else:
                item["Answer"] = "Failed to generate a response."
                answered_data.append(item)
                print("Failed to get a response.\n")

        except Exception as e:
            item["Answer"] = f"Error: {str(e)}"
            answered_data.append(item)
            print(f"Error occurred: {e}\n")

        # Avoid hitting rate limits
        time.sleep(0.3)
    return parsed_questions


# Main function to coordinate the process
def process_questions(input_file, output_file, model):
    # Step 1: Load JSON data from the input file
    with open(input_file, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # Step 2: Generate answers
    qa_pair = generate_answers(input_data["qs"], model)

    # Save the results to a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(qa_pair, f, ensure_ascii=False, indent=4)
    print(f"Answers saved to {output_file}")
