import json
import numpy as np


def select_data_by_category(data, select_per_model, select_per_category, categories):
    selected_data = []
    rest_data = []
    category_bins = {category: [] for category in categories}
    for d in data:
        category_bins[d["category"]].append(d)
    for category in categories:
        if len(category_bins[category]) <= select_per_category:
            selected_data.extend(category_bins[category])
        else:
            idxs = np.random.choice(len(category_bins[category]), select_per_category, replace=False)
            selected_data.extend([category_bins[category][idx] for idx in idxs])
            rest_data.extend(
                [category_bins[category][idx] for idx in range(len(category_bins[category])) if idx not in idxs])

    select_from_rest = select_per_model - len(selected_data)
    if select_from_rest > 0:
        selected_data.extend(np.random.choice(rest_data, select_from_rest, replace=False))
    return selected_data


if __name__ == "__main__":
    classified_categories = ["only/standard treatment", "no treatment", "inevitable side effect",
                             "causal misattribution", "underestimate risk", "no symptoms means no disease"]
    all_categories = classified_categories + ["other"]

    ### TODO: select number of examples per model based on your need
    select_per_model = 266

    ### TODO: in our example, the first 76 examples are fixed examples (annotated by physicians in the first batch)
    fixed_data_count = 76

    ### TODO: change your supported models if you want to
    supported_models = ["gpt-4o", "claude-3.5-sonnet", "gemini-1.5-pro"]

    select_per_category = select_per_model // len(all_categories)
    all_data = json.load(open("data/all_data_raw_generated.json"))[fixed_data_count:]
    filtered_data = all_data[:fixed_data_count]

    generated_data = {
        model_tag: [d for d in all_data if d["from_model"] == model_tag] for model_tag in supported_models
    }
    for model, data in generated_data.items():
        print(f"Model: {model}")
        print(f"Number of examples: {len(data)}")
        for d in data:
            d["from_model"] = model
            if d["category"] not in classified_categories:
                d['category'] = "other"
        print("Category distribution:")
        category_counts = {category: sum(1 for d in data if d["category"] == category) for category in
                           all_categories}
        print(json.dumps(category_counts, indent=2))
        filtered_data.extend(select_data_by_category(data, select_per_model, select_per_category, all_categories))

    filtered_data = sorted(filtered_data, key=lambda x: (x["source_row"], x["from_model"]))
    for i, d in enumerate(filtered_data):
        d["raw_QID"] = i
    with open("data/filter.json", "w") as f:
        json.dump(filtered_data, f, indent=2)
    print("Category distribution:")
    category_counts = {category: sum(1 for d in filtered_data if d["category"] == category) for category in
                       all_categories}
    print(json.dumps(category_counts, indent=2))
