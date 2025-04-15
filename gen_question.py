import json
import re
import numpy as np
import dspy
from key import OPENAI_API_KEY

json_pattern = r'\{\n[^{}]*\}'
system_msg = (
    "You are a helpful and knowledgeable medical assistant."
)

# Define cancer types to generate questions for
cancer_types = [
    # Common cancer types
    "lung cancer",  # 肺癌
    "breast cancer",  # 乳腺癌
    "colorectal cancer",  # 结直肠癌
    "prostate cancer",  # 前列腺癌
    "liver cancer",  # 肝癌
    "pancreatic cancer",  # 胰腺癌
    "stomach cancer",  # 胃癌
    "cervical cancer",  # 宫颈癌
    "esophageal cancer",  # 食管癌
    "skin cancer",  # 皮肤癌 (melanoma and non-melanoma)

    # Less common cancer types
    "thyroid cancer",  # 甲状腺癌
    "kidney cancer",  # 肾癌
    "bladder cancer",  # 膀胱癌
    "ovarian cancer",  # 卵巢癌
    "testicular cancer",  # 睾丸癌
    "bone cancer",  # 骨癌
    "brain cancer",  # 脑癌
    "gallbladder cancer",  # 胆囊癌
    "salivary gland cancer",  # 唾液腺癌
    "adrenal gland cancer",  # 肾上腺癌
    "non-Hodgkin's lymphoma",

    # Rare cancer types
    "sarcomas",  # 肉瘤 (soft tissue and bone)
    "neuroendocrine tumors",  # 神经内分泌肿瘤
    "mesothelioma",  # 间皮瘤
    "eye cancer",  # 眼癌 (e.g., retinoblastoma, ocular melanoma)
    "Merkel cell carcinoma",  # 默克尔细胞癌
    "small bowel cancer",  # 小肠癌
    "vulvar cancer",  # 外阴癌
    "penile cancer",  # 阴茎癌
    "nasopharyngeal cancer",  # 鼻咽癌
    "Hodgkin lymphoma"  # 霍奇金淋巴瘤
]


# Function to load example questions and assumptions from a JSON file
def load_examples_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    return examples


# Function to save responses to a JSON file
def save_responses_to_json(responses, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=4)
    print(f"Responses have been saved to '{output_file}'.")


# Function to generate prompts for GPT to create similar cancer-related questions
def generate_prompt_based_on_examples(examples, neg_examples, question_info, example_size=6, sample_size=3,
                                      bad_fixed_size=4):
    # Build a single prompt containing all examples
    good_examples_str = ""
    good_sample_size = min(example_size, len(examples))
    for example in np.random.choice(examples, good_sample_size):
        question = example["example_question"]
        assumption = example["example_assumption"]
        good_examples_str += json.dumps(
            {
                "Question": question,
                "Wrong Assumption": assumption
            }, indent=2
        ) + "\n"

    bad_examples_str = ""
    bad_sample_size = min(example_size, len(neg_examples)) - bad_fixed_size
    for example in neg_examples[:bad_fixed_size] + list(np.random.choice(neg_examples, bad_sample_size)):
        question = example["example_question"]
        reason = example["reason"]
        bad_examples_str += json.dumps(
            {
                "Question": question,
                "Bad Reason": reason
            }, indent=2
        ) + "\n"

    prompt = (
        f"You are asked to generate medical patient questions but with false presuppositions."
        f"Here are some VALID example questions and their false presuppositions:\n"
        f"{good_examples_str}\n"
        f"Here are some INVALID example questions and the reasons why are they NOT valid:\n"
        f"{bad_examples_str}\n"
        f"Please generate {sample_size} other VALID questions with wrong assumptions related to {question_info}, using the same JSON format as above.\n"
        f"The questions should NOT use similar presuppositions as in the valid examples.\n"
        f"The questions should NOT directly state or question the presuppositions but instead reflect the patient's misunderstanding in a more subtle way.\n"
        f"The false presuppositions should be hard and can trigger LLM error.\n"
        f"The presuppositions should be IMPLICIT, and the question should be based on that hidden presuppositions, as shown in the VALID examples above."
    )
    return prompt


# Function to process examples and generate new cancer-related questions
def generate_similar_questions(examples, neg_examples, question_info, model):
    """
    :param examples: list of example questions and assumptions
    :param neg_examples: list of negative example questions and reasons
    :param question_info: cancer type(in random mode) or relevant myth info(in myth mode) to generate questions for
    :return: a list of generated 3 questions
    """

    # Generate prompts based on examples
    prompt = generate_prompt_based_on_examples(examples, neg_examples, question_info)

    print("Generated Questions")
    # Call GPT 4o to generate similar questions based on the prompt
    response = model(prompt)[0]
    if response:
        questions = [json.loads(match) for match in re.findall(json_pattern, response)]
        for q in questions:
            print(json.dumps(q, indent=2))
        return {"prompt": prompt, "response": response, "qs": questions}
    else:
        print("Failed to get a response.\n")
        return None
