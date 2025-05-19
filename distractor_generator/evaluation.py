import json
import random
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download("punkt")

# BLEU smoothing
smoothie = SmoothingFunction().method4

# Load dataset
with open("distractor_generator/dataset/distractor_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Sample 10
random.seed(42)
number_of_samples = 200
model_name = "final-model-500"
sampled_data = random.sample(dataset, number_of_samples)

# Load model
model_path = f"distractor_generator/model_final/{model_name}"
tokenizer = T5TokenizerFast.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

print(f"🔍 Testing {number_of_samples} Random Examples:\n")

# Helpers
def rough_overlap_score(reference, prediction):
    ref_tokens = set(reference.lower().split())
    pred_tokens = set(prediction.lower().split())
    return len(ref_tokens & pred_tokens) / max(len(ref_tokens), 1)

def generate_valid_distractors(input_text, correct_answer, tokenizer, model, max_tries=5):
    best_output = None
    best_score = -1
    best_distractors = []

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

        if "Distractor:" in raw_output:
            distractors = raw_output.replace("Distractor:", "").split(",")
            distractors = [d.strip() for d in distractors]
            unique = list(dict.fromkeys(distractors))
            filtered = [d for d in unique if d.lower() != correct_answer.lower()]

            score = len(filtered)
            if score > best_score:
                best_score = score
                best_output = "Distractor: " + ", ".join(filtered[:3]) + (" <fallback>" if score < 3 else "")
                best_distractors = filtered[:3]

            if score >= 3:
                return best_output, best_distractors, True

    return best_output or "Distractor: <placeholder>, <placeholder>, <placeholder>", best_distractors, False

# Store report
report_data = []

# Run predictions and evaluation
for i, example in enumerate(sampled_data, 1):
    input_text = example["input"]
    expected_output = example["output"]
    correct_answer = input_text.split("with answer:")[-1].strip()

    expected_list = [x.strip() for x in expected_output.replace("Distractor:", "").split(",")]

    # Generate
    generated_output, predicted_list, is_valid = generate_valid_distractors(
        input_text, correct_answer, tokenizer, model
    )

    # Scores
    bleu = sentence_bleu([expected_list], predicted_list, smoothing_function=smoothie)
    rough = rough_overlap_score(expected_output, generated_output)

    # Show
    print(f"{i}. Input     : {input_text}")
    print(f"   Expected : {expected_output}")
    print(f"   Predicted: {generated_output}")
    print(f"   BLEU     : {bleu:.2f}")
    print(f"   Rough    : {rough:.2f}")
    print("-" * 80)

    # Save
    report_data.append({
        "input": input_text,
        "expected": expected_output,
        "predicted": generated_output,
        "expected_list": expected_list,
        "predicted_list": predicted_list,
        "correct_answer": correct_answer,
        "bleu_score": round(bleu, 4),
        "rough_score": round(rough, 4),
    })

# Save report
with open(f"distractor_generator/eval_report_{model_name}.json", "w", encoding="utf-8") as f:
    json.dump(report_data, f, indent=4)

print(f"✅ Evaluation complete. Results saved to 'eval_report_{model_name}.json'")
