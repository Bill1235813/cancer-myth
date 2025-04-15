import json
import pandas as pd
import os

# Mapping from model names to their corresponding JSON result files
model_to_json = {
    "Claude-3.5-Sonnet": "evaluation_results_anthropic_claude-3-5-sonnet-20240620.json",
    "DeepSeek-V3": "evaluation_results_deepseek_deepseek-chat.json",
    "DeepSeek-R1": "evaluation_results_deepseek_deepseek-reasoner.json",
    "Gemini-1.5-Pro": "evaluation_results_gemini_gemini-1.5-pro.json",
    "GPT-3.5": "evaluation_results_openai_gpt-3.5-turbo.json",
    "GPT-4-Turbo": "evaluation_results_openai_gpt-4-turbo.json",
    "GPT-4o": "evaluation_results_openai_gpt-4o.json",
    "GPT-4o (MDAgents)": "evaluation_results_openai_gpt-4o-mdagents.json",
}

# Load model output data
model_data = {}
for model, file in model_to_json.items():
    output_path = f"output/{file}"
    if os.path.exists(output_path):
        model_data[model] = json.load(open(output_path, encoding="utf-8"))
    else:
        print(f"Warning: File not found for model {model}: {output_path}")
        model_data[model] = []

# Load example data
data = json.load(open("data/all_data.json", encoding="utf-8"))

# Combine and structure the full dataset
data_indexed = [
    {
        "example_question": d["example_question"],
        "example_assumption": d["example_assumption"],
        "source_row": d["source_row"],
        "source_type": d["source_type"],
        "source_info": d["source_info"],
        "category": d["category"],
        "from_model": d.get("from_model", "gpt-4o"),
        "answers": {},
        "scores": {},
        "QID": i
    }
    for i, d in enumerate(data)
]

# Match each question with corresponding model answers and scores
for d in data_indexed:
    matched = False
    for i, ans_d in enumerate(model_data["GPT-4o"]):
        if d["example_question"] == ans_d["question"]:
            for model, ans_data in model_data.items():
                if i < len(ans_data) and ans_data[i] is not None:
                    d["answers"][model] = ans_data[i]["model_answer"]
                    d["scores"][model] = ans_data[i]["evaluation"][0]["Sharpness"]
            matched = True
            break
    if not matched:
        print("Not found:", d["example_question"])

# Directly combine all indexed data (no filtering)
final_data = data_indexed
print("Total entries:", len(final_data))

# Save result
with open("all_data.json", "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)
