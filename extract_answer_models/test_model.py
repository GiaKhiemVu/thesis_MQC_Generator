import json
import random
from transformers import T5TokenizerFast, T5ForConditionalGeneration

# Load dataset
with open("extract_answer_models/dataset/cleaned_data.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Shuffle and take 10 random samples
random.seed(42)
sampled_data = random.sample(dataset, 10)

# Load model and tokenizer
model_path = "extract_answer_models/model_final/final-model-2000"
tokenizer = T5TokenizerFast.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

print("🔍 Testing 10 Random Examples:\n")

# Run predictions
for i, example in enumerate(sampled_data, 1):
    input_text = example["input"]
    expected_output = example["output"]

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)

    # Generate output
    outputs = model.generate(**inputs, max_length=64)

    # Decode
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Show results
    print(f"{i}. Input     : {input_text}")
    print(f"   Expected : {expected_output}")
    print(f"   Predicted: {decoded_output}")
    print("-" * 80)
