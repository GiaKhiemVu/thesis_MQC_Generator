import json

print("Preprocessing for distractor generation...")
# Load the original dataset
with open("original_dataset.json", "r", encoding="utf-8") as f:
    original_data = json.load(f)

# Transform the data
transformed_data = []
for entry in original_data:
    question = entry["question"]
    answer = entry["correct_answer"]
    distractors = entry["distractors"]
    question_input = question.replace("<BLANK>", "_")
    new_entry = {
        "input": f"Generate 3 distractors no duplicated for question: {question_input} with answer: {answer}",
        "output": f"Distractor: {', '.join(distractors)}"
    }
    transformed_data.append(new_entry)

print(f"Processed dataset size: {len(transformed_data)}")
# Save the new file
with open("distractor_generator/dataset/distractor_dataset.json", "w", encoding="utf-8") as f:
    json.dump(transformed_data, f, indent=4)
