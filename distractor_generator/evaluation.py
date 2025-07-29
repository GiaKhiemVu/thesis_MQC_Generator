import json
import random
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import torch
nltk.download("punkt")

# BLEU smoothing
smoothie = SmoothingFunction().method4

# Load dataset
with open("distractor_generator/dataset/distractor_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Sample 10
random.seed(42)
number_of_samples = 200
model_name = "final-model-2999"
sampled_data = random.sample(dataset, number_of_samples)

# Load model
model_path = f"distractor_generator/model_final_base/{model_name}"
tokenizer = T5TokenizerFast.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

print(f"🔍 Testing {number_of_samples} Random Examples:\n")

# Helpers
def rough_overlap_score(reference, prediction):
    ref_tokens = set(reference.lower().split())
    pred_tokens = set(prediction.lower().split())
    return len(ref_tokens & pred_tokens) / max(len(ref_tokens), 1)

def generate_valid_distractors(input_text, correct_answer, tokenizer, model, max_attempts=20):
    if not correct_answer:
        raise ValueError("correct_answer must be provided to validate distractors.")

    correct_answer_lower = correct_answer.lower()
    collected_distractors = set()
    raw_outputs = []
    print(f"🔄 {input_text}")
    for attempt in range(max_attempts):
        seed = random.randint(0, 100_000_000)
        torch.manual_seed(seed)

        inputs = tokenizer(input_text, return_tensors="pt", padding=True)

        outputs = model.generate(
            **inputs,
            max_length=64,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.5,
            repetition_penalty=2.0,
            num_return_sequences=1
        )

        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        raw_outputs.append(raw_output)
        
        distractor_list = [d.strip() for d in raw_output.replace("Distractors:", "").replace("Distractor:", "").split(",")]

        for d in distractor_list:
            d_lower = d.lower()
            if d_lower == correct_answer_lower:
                continue
            if d_lower in collected_distractors:
                continue
            if correct_answer_lower in d_lower or d_lower in correct_answer_lower:
                continue

            collected_distractors.add(d_lower)

            if len(collected_distractors) == 3:
                print(f"✅ Collected 3 valid distractors in attempt {attempt + 1}")
                distractor_str = f"Distractors: {', '.join(collected_distractors)}"
                return {
                    "input": input_text,
                    "output": distractor_str,
                    "distractors": list(collected_distractors),
                    "valid": True
                }

        print(f"Attempt {attempt + 1} → Valid so far: {list(collected_distractors)}")

    print("⚠️ Max attempts reached. Returning what was collected.")
    distractor_str = f"Distractors: {', '.join(collected_distractors)}"
    return {
        "input": input_text,
        "output": distractor_str,
        "distractors": f"{list(collected_distractors)}",
        "valid": False
    }

report_data = []

for i, example in enumerate(sampled_data, 1):
    input_text = example["input"]
    expected_output = example["output"]
    correct_answer = input_text.split("with answer:")[-1].strip()
    expected_list = [x.strip() for x in expected_output.replace("Distractor:", "").split(",")]

    # Generate distractors
    result = generate_valid_distractors(input_text, correct_answer, tokenizer, model)

    predicted_output = result["output"]
    predicted_list = result["distractors"]
    is_valid = result["valid"]
    print(expected_list, predicted_list)
    # Calculate scores
    bleu = sentence_bleu([expected_list], predicted_list, smoothing_function=smoothie)
    rough = rough_overlap_score(expected_output, predicted_output)


    # Log
    print(f"{i}. Input     : {input_text}")
    print(f"   Expected : {expected_output}")
    print(f"   Predicted: {predicted_output}")
    print(f"   BLEU     : {bleu:.2f}")
    print(f"   Rough    : {rough:.2f}")
    print("-" * 80)

    # Collect results
    report_data.append({
        "input": input_text,
        "expected": expected_output,
        "predicted": predicted_output,
        "expected_list": expected_output,
        "predicted_list": predicted_list,
        "correct_answer": correct_answer,
        "bleu_score": round(bleu, 4),
        "rough_score": round(rough, 4),
        "valid": is_valid
    })

# Save the evaluation report
with open(f"distractor_generator/eval_report_{model_name}.json", "w", encoding="utf-8") as f:
    json.dump(report_data, f, indent=4)

print(f"✅ Evaluation complete. Results saved to 'eval_report_{model_name}.json'")
