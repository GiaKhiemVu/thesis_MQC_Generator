import json

# Load raw dataset
with open("original_dataset.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

cleaned_data = []
seen_questions = set()

for entry in raw_data:
    question = entry.get("question", "").strip()
    correct_answer = entry.get("correct_answer", "").strip()
    distractors = entry.get("distractors", [])

    # Skip if any essential part is missing or distractors are not a valid list
    if not question or not correct_answer or not isinstance(distractors, list):
        continue

    # Clean and validate distractors
    cleaned_distractors = [d.strip() for d in distractors if isinstance(d, str) and d.strip()]
    if len(cleaned_distractors) != 3:
        continue

    # Skip duplicate questions
    if question in seen_questions:
        continue
    seen_questions.add(question)

    # Append cleaned entry
    cleaned_data.append({
        "question": question,
        "correct_answer": correct_answer,
        "distractors": cleaned_distractors
    })

# Save cleaned dataset
with open("original_dataset.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

print(f"✅ Cleaned dataset saved. Total entries: {len(cleaned_data)}")
