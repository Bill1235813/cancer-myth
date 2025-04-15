from transformers import AutoTokenizer
from huggingface_hub import login
import numpy as np
import pandas as pd
import re

# Replace 'your_hugging_face_token' with your actual token
login(token="your_hugging_face_token")


def markdown_to_text(markdown_str):
    # Remove markdown specific syntax (bold, italics, headers, etc.)
    plain_text = re.sub(r'(\*\*|\*|_|`|~~|#)', '', markdown_str)
    # Remove extra newlines
    plain_text = re.sub(r'\n\n+', '\n\n', plain_text)
    return plain_text.strip()


filename = "../data/human_answer_with_models.csv"
column_names = ["Edited human answer", "GPT-4-Turbo", "LLaMa-3.1-Instruct 405B",
                "Gemini-1.5 Pro"]
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

csv_data = pd.read_csv(filename, header=0)

for column_name in column_names:
    if column_name in csv_data:
        csv_data[column_name] = [markdown_to_text(ans_str) for ans_str in csv_data[column_name]]

        str_len = [len(tok.encode(ans_str)) for ans_str in csv_data[column_name]]
        print("Model name:", column_name)
        print("Tokens:", np.mean(str_len), max(str_len), min(str_len))

        str_word_len = [len(ans_str.split()) for ans_str in csv_data[column_name]]
        print("Words:", np.mean(str_word_len), max(str_word_len), min(str_word_len))

        str_para_len = [len(ans_str.split("\n\n")) for ans_str in csv_data[column_name]]
        print("Paragraphs:", np.mean(str_para_len), max(str_para_len), min(str_para_len))
        print()
