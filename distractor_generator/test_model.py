import json
import random
from transformers import T5TokenizerFast, T5ForConditionalGeneration

# Load dataset
with open("distractor_generator/dataset/distractor_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Shuffle and take 10 random samples
random.seed(42)
sampled_data = random.sample(dataset, 10)

# Load model and tokenizer
model_path = "distractor_generator\model_train_base\model-extract-2999\checkpoint-6000"
tokenizer = T5TokenizerFast.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

print("🔍 Testing 10 Random Examples:\n")

# Post-processing function to remove duplicates and clean formatting
def clean_distractor_output(output_str):
    if "Distractor:" in output_str:
        items = output_str.replace("Distractor:", "").split(",")
        unique_items = list(dict.fromkeys([i.strip() for i in items]))
        return "Distractor: " + ", ".join(unique_items[:3])
    return output_str.strip()

def generate_valid_distractors(input_text, correct_answer, tokenizer, model, max_tries=10):
    best_output = None
    best_score = -1

    for _ in range(max_tries):
        inputs = tokenizer(input_text, return_tensors="pt", padding=True)

        outputs = model.generate(
            **inputs,
            max_length=64,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.2,
            repetition_penalty=2.0,
            num_return_sequences=1
        )

        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Distractors:" in raw_output:
            distractors = raw_output.replace("Distractors:", "").split(",")
            distractors = [d.strip() for d in distractors]
            unique = list(dict.fromkeys(distractors))
            filtered = [d for d in unique if d.lower() != correct_answer.lower()]

            score = len(filtered)
            if score > best_score:
                best_score = score
                best_output = "Distractors: " + ", ".join(filtered[:3]) + (" <fallback>" if score < 3 else "")

            if score >= 3:
                return best_output, True  # Fully valid set found

    return best_output or "Distractors: <placeholder>, <placeholder>, <placeholder>", False


# Run predictions
for i, example in enumerate(sampled_data, 1):
    input_text = example["input"]
    expected_output = example["output"]
    correct_answer = input_text.split("with answer:")[-1].strip()

    # Try to generate valid distractors
    generated_output, is_valid = generate_valid_distractors(input_text, correct_answer, tokenizer, model)

    # Display result
    match = "✅" if is_valid else "❌"
    print(f"{i}. Input     : {input_text}")
    print(f"   Expected : {expected_output}")
    print(f"   Predicted: {generated_output} {match}")
    if not is_valid:
        print("   ⚠️  Could not generate 3 valid distractors after retries.")
    print("-" * 80)