import json
import os
import matplotlib.pyplot as plt
import pandas as pd

# File paths
file_paths = {
    "Model-500": "distractor_generator/eval_report_final-model-500.json",
    "Model-1000": "distractor_generator/eval_report_final-model-1000.json",
    "Model-1500": "distractor_generator/eval_report_final-model-1500.json",
    "Model-2000": "distractor_generator/eval_report_final-model-2000.json"
}

# Data containers
bleu_averages = {}
rough_averages = {}

# Load and process each file
for model_name, file_path in file_paths.items():
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        bleu_scores = [entry["bleu_score"] for entry in data]
        rough_scores = [entry["rough_score"] for entry in data]
        bleu_averages[model_name] = sum(bleu_scores) / len(bleu_scores)
        rough_averages[model_name] = sum(rough_scores) / len(rough_scores)

# ✅ Now safe to extract values
labels = list(file_paths.keys())
bleu_values = [bleu_averages[label] for label in labels]
rough_values = [rough_averages[label] for label in labels]

x = range(len(labels))  # x-axis locations
width = 0.35  # width of the bars

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar([i - width/2 for i in x], bleu_values, width, label='BLEU Score')
bars2 = ax.bar([i + width/2 for i in x], rough_values, width, label='Rough Score')

# Add some text for labels, title and axes ticks
ax.set_xlabel('Model (Training Records)')
ax.set_ylabel('Average Score')
ax.set_title('BLEU vs Rough Score by Model')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

# Save the bar chart as PDF
bar_chart_path = "evaluation_data/compare_output_distractor_generator_models.pdf"
plt.tight_layout()
plt.savefig(bar_chart_path)
plt.show()
