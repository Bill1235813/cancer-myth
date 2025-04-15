import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pprint

data = pd.read_csv("survey_results.csv")
qid_list = json.load(open("../questionnaire/full_survey.json"))
model_to_ids = {
    "Gemini-1.5 Pro": [],
    "LLaMa-3.1-Instruct 405B": [],
    "GPT-4-Turbo": [],
    "Edited human answer": [],
}
ids_to_model = {}
count = 0
for _, model in qid_list.items():
    for m in model:
        model_to_ids[m].append(count)
        ids_to_model[count] = m
        count += 1

pprint.pprint(ids_to_model)

qid_ans = [[] for _ in range(len(qid_list))]
para_ans = [[] for _ in range(len(qid_list) * 4)]
para_ans_all = [[] for _ in range(len(qid_list) * 4)]
label_count = {
    "Overly generic": [],
    "Bad or inappropriate advice": [],
    "Not following guidelines": [],
    "Not realistic for patient": [],
    "Factually incorrect": []
}

print("Q count", len(qid_ans))
print("Model count", len(para_ans))

id_to_score = {}

### Parse the data
id_count = 0
id_inner = 0
para_ans_count = 0
para_ans_nocount = 0
for key in data:
    if key.startswith("QID") and int(key.split("QID")[-1]) % 100 <= 98:
        if int(key.split("QID")[-1]) % 5 == 0 or int(key.split("QID")[-1]) % 100 == 98:
            try:
                for d in data[key][3:]:
                    if isinstance(d, str):
                        for l in d.strip().split(","):
                            if l in label_count:
                                label_count[l].append(ids_to_model[int(key.split("QID")[-1]) // 100])
                                if ids_to_model[
                                    int(key.split("QID")[-1]) // 100] == "GPT-4-Turbo" and l == "Factually incorrect":
                                    print("F", key)
                                elif ids_to_model[int(key.split("QID")[
                                                          -1]) // 100] == "GPT-4-Turbo" and l == "Not realistic for patient":
                                    print("R", key)
            except:
                print(data[key][3:])
        if int(key.split("QID")[-1]) % 5 == 4 or int(key.split("QID")[-1]) % 100 == 98:
            try:
                id_to_score[key] = list(int(i) for i in data[key][3:])
            except:
                id_to_score[key] = "No advice"
                # print(key, int(key.split("QID")[-1]) % 100, data[key][3:])
            if int(key.split("QID")[-1]) % 100 == 98:
                qid_ans[id_count].append(np.mean(id_to_score[key]))
                id_inner += 1
                if id_inner % 4 == 0:
                    id_count += 1
            else:
                if id_to_score[key] == "No advice":
                    para_ans_nocount += 1
                else:
                    para_ans_count += 1
                    para_ans[id_inner].append(np.mean(id_to_score[key]))
                    para_ans_all[id_inner] += id_to_score[key]

print(para_ans_count, para_ans_nocount, len(para_ans[0]))

### Prepare the data for plotting
model_name_map = {
    "Gemini-1.5 Pro": "Gemini-1.5-Pro",
    "LLaMa-3.1-Instruct 405B": "LLaMa-3.1-Instruct-405B",
    "GPT-4-Turbo": "GPT-4-Turbo",
    "Edited human answer": "Social Workers",
}
model_score = {
    "Gemini-1.5-Pro": [],
    "LLaMa-3.1-Instruct-405B": [],
    "GPT-4-Turbo": [],
    "Social Workers": [],
}
model_para_score = {
    "Gemini-1.5-Pro": [],
    "LLaMa-3.1-Instruct-405B": [],
    "GPT-4-Turbo": [],
    "Social Workers": [],
}
model_diff = {
    "Gemini-1.5-Pro": [],
    "LLaMa-3.1-Instruct-405B": [],
    "GPT-4-Turbo": [],
    "Social Workers": [],
}
for id, qid in enumerate(qid_list):
    for mid, (model, score) in enumerate(zip(qid_list[qid], qid_ans[id])):
        model_score[model_name_map[model]].append(score)
        model_para_score[model_name_map[model]] += para_ans_all[id * 4 + mid]

for model in model_para_score:
    model_para_score[model] = np.array(model_para_score[model])
#     print(model, len(score), np.mean(score))

for key, score_list in id_to_score.items():
    if score_list != "No advice":
        model_diff[model_name_map[ids_to_model[int(key.split("QID")[-1]) // 100]]].append(score_list)
for model in model_diff:
    model_diff[model] = np.array(model_diff[model])

# ### Plot harmful data
# data = {
#     'Score': list(label_count.keys()) * len(model_to_ids),
#     'Count': list(itertools.chain.from_iterable(
#         [[label_list.count(model) for label, label_list in label_count.items()]
#          for model, _ in model_to_ids.items()])),
#     'Model name': list(itertools.chain.from_iterable([[key] * 5 for key in model_to_ids]))
# }
# df = pd.DataFrame(data)
# plt.figure(figsize=(6, 6))
# sns.barplot(data=df, x='Score', y='Count', hue='Model name', palette="Set2")
# plt.title('Distribution of "Harmful Category" from the Advice')
# plt.xlabel('Category')
# plt.ylabel('Count')
# plt.xticks(rotation=30)
# plt.legend()
# plt.tight_layout()
# plt.savefig("category_count.pdf")
# plt.savefig("category_count.png", dpi=400)
#
# ### Plot advice distribution
# data = {
#     'Score': list(["1-2", "2-3", "3-4", "4-5"]) * len(model_para_score),
#     'Count': list(itertools.chain.from_iterable(
#         [[len(score[(score >= 1) & (score <= 2)])] + [len(score[(score > i) & (score <= i + 1)]) for i in range(2, 5)]
#          for key, score in model_para_score.items()])),
#     'Model name': list(itertools.chain.from_iterable([[key] * 4 for key in model_para_score]))
# }
# df = pd.DataFrame(data)
# plt.figure(figsize=(8, 6))
# sns.barplot(data=df, x='Score', y='Count', hue='Model name', palette="Set2")
# plt.title('Score Distribution of Advice (Paragraph)', fontsize=18)
# plt.xlabel('Score', fontsize=18)
# plt.ylabel('Count', fontsize=18)
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# plt.legend(fontsize=18)
# plt.tight_layout()
# plt.savefig("para_count.pdf")
# plt.savefig("para_count.png", dpi=400)

### Create combined plot with two subplots and single legend
plt.rcParams.update({'font.size': 16})  # Global font size increase

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# First subplot: Harmful Category Distribution
data1 = {
    'Score': list(label_count.keys()) * len(model_to_ids),
    'Count': list(itertools.chain.from_iterable(
        [[label_list.count(model) for label, label_list in label_count.items()]
         for model, _ in model_to_ids.items()])),
    'Model name': list(itertools.chain.from_iterable([[key] * 5 for key in model_to_ids]))
}
df1 = pd.DataFrame(data1)

# Plot first subplot
sns.barplot(data=df1, x='Score', y='Count', hue='Model name', palette="deep", ax=ax1)
# ax1.set_title('Distribution of "Harmful Category"', fontsize=18)
ax1.set_xlabel('', fontsize=18)
ax1.set_ylabel('Count', fontsize=18)
ax1.set_xticklabels(
    ["Overly \n generic", "Unsuitable \n advice", "Off \n guidelines", "Unrealistic", "Factually \n incorrect"],
    fontsize=16)
ax1.tick_params(axis='y', labelsize=16)
ax1.get_legend().remove()  # Remove the first legend

# Second subplot: Advice Score Distribution
# data2 = {
#     'Score': list(["1-2", "2-3", "3-4", "4-5"]) * len(model_para_score),
#     'Count': list(itertools.chain.from_iterable(
#         [[len(score[(score >= 1) & (score <= 2)])] + [len(score[(score > i) & (score <= i + 1)]) for i in range(2, 5)]
#          for key, score in model_para_score.items()])),
#     'Model name': list(itertools.chain.from_iterable([[key] * 4 for key in model_para_score]))
# }
data2 = {
    'Score': list(["1", "2", "3", "4", "5"]) * len(model_para_score),
    'Count': list(itertools.chain.from_iterable(
        [[len(score[(score == i)]) for i in range(1, 6)]
         for key, score in model_para_score.items()])),
    'Model name': list(itertools.chain.from_iterable([[key] * 5 for key in model_para_score]))
}
df2 = pd.DataFrame(data2)

# Plot second subplot
sns.barplot(data=df2, x='Score', y='Count', hue='Model name', palette="deep", ax=ax2)
# ax2.set_title('Score Distribution of Advice (Paragraph)', fontsize=18)
ax2.set_xlabel('Paragraph Score', fontsize=18)
ax2.set_ylabel('Count', fontsize=18)
ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=16)
ax2.tick_params(axis='y', labelsize=16)
ax2.get_legend().remove()  # Remove the second legend

# Create a single legend for the entire figure
handles, labels = ax2.get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0),
                    ncol=len(labels))

# Explicitly set legend font sizes
plt.setp(legend.get_title(), fontsize=18)  # Legend title
plt.setp(legend.get_texts(), fontsize=18)  # Legend text

# Adjust layout and save
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)  # Make room for the legend
plt.savefig("combined_prelim.pdf", bbox_inches='tight', dpi=500)
plt.savefig("combined_prelim.png", dpi=400, bbox_inches='tight')

### Plot agreement difference
data = {
    'Score': list(["0", "1", "2", "3", "4"]) * len(model_diff),
    'Count': list(itertools.chain.from_iterable(
        [[((np.max(score, axis=1) - np.min(score, axis=1)) == i).sum() for i in range(5)]
         for key, score in model_diff.items()])),
    'Model name': list(itertools.chain.from_iterable([[key] * 5 for key in model_para_score]))
}
df = pd.DataFrame(data)
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Score', y='Count', hue='Model name', palette=sns.color_palette("deep", 4))
plt.title('Distribution of Agreement between Fellows', fontsize=18)
plt.xlabel('Max Score Difference', fontsize=18)
plt.ylabel('Count', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig("agreement_count.pdf")
plt.savefig("agreement_count.png", dpi=400)

# Overall score per model
print("Overall score per question:")
for model, score in model_score.items():
    print("%s: %.2f" % (model, np.mean(score)))

# Per person score: Doctor1, Doctor2, Doctor3
for model, score in model_diff.items():
    print("%s: \n\tDoctor1 - %.2f, Doctor2 - %.2f, Doctor3 - %.2f" % ((model,) + tuple(np.mean(score, axis=0))))
