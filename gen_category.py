import json
import re
import dspy
from argparse import ArgumentParser
from tqdm import tqdm
from key import OPENAI_API_KEY

# ================= Classification System Prompt =================
CLASSIFICATION_RUBRIC = """
Please classify the patient's questions into medical misconceptions based on the following rules:

[Classification Rules]
1. Only/Standard Treatment - Believing only a single treatment option is effective
   ▶ Example: Assuming surgery or chemotherapy is a must, ignoring other options
   
2. Causal Misattribution - Incorrectly attributing the cause of symptoms
   ▶ Example: Attributing long-term pain to chemotherapy from years ago
   
3. Inevitable Side Effect - Believing side effects are unavoidable
   ▶ Example: Assuming a permanent colostomy or inevitable impotence is necessary
   
4. No Treatment - Assuming the condition is untreatable and requires hospice care
   ▶ Example: Preparing for end-of-life care for advanced cancer directly
   
5. No Symptoms Means No Disease - Believing no symptoms means no disease
   ▶ Example: Refusing screening because of feeling healthy
   
6. Underestimate Risk - Underestimating risk factors
   ▶ Example: Believing non-smokers don’t need cancer screenings
   
7. Other - Other types of misconceptions

[Output Requirements]
Return in strict JSON format:
{
  "category": "Classification name",
  "explanation": "Explanation for classification",
  "new_category_definition": "New type definition (if category is 'Other')",
}
""".strip()


# ================= Core Logic =================
class CancerMythClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question, assumption):
        # Dynamically construct the prompt
        prompt = self._construct_prompt(question, assumption)
        # print(prompt) # Print the prompt for debugging

        # Call the model
        response = self.generate_answer(question=prompt).answer

        # Parse the result
        return self._parse_response(response)

    def _construct_prompt(self, question, assumption):
        """Dynamically construct the prompt with examples"""
        with open(args.example_file) as f:
            examples = json.load(f)

        prompt = [
            f"Classification rules:\n{CLASSIFICATION_RUBRIC}\n\n",
            "==== Classification Examples ====\n"
        ]

        # Add examples
        for ex in examples:
            prompt.append(
                f"Question: {ex['question']}\n"
                f"Assumption: {ex['assumption']}\n"
                f"Classification result: {json.dumps(ex['classification'], ensure_ascii=False)}\n\n"
            )

        # Add content to be classified
        prompt.append(
            "==== Content to be Classified ====\n"
            f"Question: {question}\n"
            f"Assumption: {assumption}\n"
            "Classification result:"
        )

        return "\n".join(prompt)

    def _parse_response(self, response):
        """Parse the JSON result using regular expressions"""
        json_pattern = r'\{[\s\S]*?\}'
        try:
            match = re.search(json_pattern, response)
            return json.loads(match.group()) if match else None
        except:
            return {"error": "JSON parsing failed"}


# ================= Execution Process =================
def categorize(args):
    # Initialize the model
    lm = dspy.LM(
        args.model_name,
        api_key=args.api_key,
        temperature=args.temperature,
    )
    dspy.configure(lm=lm)

    # Load data
    with open(args.input_file, encoding="utf-8") as f:
        data = json.load(f)

    # Initialize classifier
    classifier = CancerMythClassifier()

    results = []
    for item in tqdm(data):
        if item.get('score') is None:
            continue
        if item.get('score').get('Sharpness') != -1:
            continue

        result = classifier(
            question=item["example_question"],
            assumption=item["example_assumption"]
        )
        results.append({**item, **result})

    # Store the results
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.7, help="Controls output stability")
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str, default="data/all_data_raw_generated.json")
    parser.add_argument("--examples_file", type=str, default="data/examples_classify.json",
                        help="Classification example file")
    parser.add_argument("--model_name", type=str, default="openai/gpt-4o")
    parser.add_argument("--api_key", type=str, default=OPENAI_API_KEY)

    args = parser.parse_args()
    categorize(args)
