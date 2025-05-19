import json
import random
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download("punkt")  # Needed for tokenization

# BLEU smoothing function
smoothie = SmoothingFunction().method4
model_name = "final-model-500"
# Load dataset
with open("extract_answer_models/dataset/cleaned_data.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

random.seed(42)
number_of_samples = 200
sampled_data = random.sample(dataset, number_of_samples)

# Load model and tokenizer
model_path = f"extract_answer_models/model_final/{model_name}"
tokenizer = T5TokenizerFast.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

print(f"🔍 Evaluating {number_of_samples} Random Examples:\n")

# Evaluation storage
report = []

def rough_overlap_score(reference, prediction):
    ref_tokens = set(reference.lower().split())
    pred_tokens = set(prediction.lower().split())
    return len(ref_tokens & pred_tokens) / max(len(ref_tokens), 1)

# Run evaluation
for i, example in enumerate(sampled_data, 1):
    input_text = example["input"]
    expected_output = example["output"]

    # Tokenize and generate
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs, max_length=64)
    predicted_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Scores
    reference_tokens = [expected_output.lower().split()]
    candidate_tokens = predicted_output.lower().split()

    bleu = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie)
    rough = rough_overlap_score(expected_output, predicted_output)

    # Print and record
    print(f"{i}. Input     : {input_text}")
    print(f"   Expected : {expected_output}")
    print(f"   Predicted: {predicted_output}")
    print(f"   BLEU     : {bleu:.2f}")
    print(f"   Rough    : {rough:.2f}")
    print("-" * 80)

    report.append({
        "input": input_text,
        "expected": expected_output,
        "predicted": predicted_output,
        "bleu_score": round(bleu, 4),
        "rough_score": round(rough, 4)
    })

# Save results
with open(f"extract_answer_models/eval_report_{model_name}.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)

print(f"✅ Evaluation complete. Results saved to 'eval_report_{model_name}.json'")
