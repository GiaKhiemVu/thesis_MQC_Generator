import json

print("Preprocessing for extract_answer_models...")
# Load your original dataset
with open('original_dataset.json', 'r') as f:
    data = json.load(f)

# Convert each record for both formats
extract_dataset = []
cleaned_data = []

for item in data:
    original_question = item["question"]
    correct_answer = item["correct_answer"]

    # Replace <BLANK> with the correct answer for input
    question_input = original_question.replace("<BLANK>", correct_answer)

    # Format 1: extract_dataset.json
    extract_dataset.append({
        "question_input": question_input,
        "question_output": original_question,
        "answer_output": correct_answer
    })

    # Format 2: cleaned_data.json
    cleaned_input = f"Convert following sentence into fill in blank question. Sentence: {question_input}"
    question_with_underscore = original_question.replace("<BLANK>", "_")
    cleaned_output = f"question: {question_with_underscore} ||| answer: {correct_answer}"

    cleaned_data.append({
        "input": cleaned_input,
        "output": cleaned_output
    })

print(f"Extract dataset size: {len(extract_dataset)}")
# Save to extract_dataset.json
with open('extract_answer_models/dataset/extract_dataset.json', 'w') as f:
    json.dump(extract_dataset, f, indent=2)

# Save to cleaned_data.json
with open('extract_answer_models/dataset/cleaned_data.json', 'w') as f:
    json.dump(cleaned_data, f, indent=2)
