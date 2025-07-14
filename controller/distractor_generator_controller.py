from transformers import T5TokenizerFast, T5ForConditionalGeneration
import random
import torch
from helper import Helper

print("----- Initializing Distractor Generator Controller -----")
print("-- Loading Model and Tokenizer --")

model_path = "distractor_generator/model_final_base/final-model-2999"
tokenizer = T5TokenizerFast.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

print("-- Model and Tokenizer Loaded --")

def preprocess_input(input_text):
    """
    Preprocess the input text for the T5 model.
    """
    return f"Generate 3 distractors no duplicated for question: {input_text.get('question', '')} Correct answer: {input_text.get('answer', '')}"

def generate_distractors(input_text, max_attempts=20):
    input_text = Helper.convert_output_step_1_into_input_step_2(input_text)

    if not input_text.get("answer"):
        raise ValueError("correct_answer must be provided to validate distractors.")

    correct_answer_lower = input_text["answer"].lower()
    collected_distractors = set()
    raw_outputs = []

    for attempt in range(max_attempts):
        seed = random.randint(0, 100_000_000)
        torch.manual_seed(seed)

        preprocessed_input = preprocess_input(input_text)
        inputs = tokenizer(preprocessed_input, return_tensors="pt", padding=True)

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
                return {
                    "input": preprocessed_input,
                    "output": str(collected_distractors),
                    "distractors": list(collected_distractors)
                }

        print(f"Attempt {attempt + 1} → Valid so far: {list(collected_distractors)}")

    print("⚠️ Max attempts reached. Returning what was collected.")
    return {
        "input": preprocessed_input,
        "output": raw_outputs,
        "distractors": list(collected_distractors)
    }

def generate_distractors_with_model(input_text, model_name, max_attempts=20):
    # Load model and tokenizer
    model_path_input = Helper.get_distractor_model_path(model_name)
    tokenizer_input = T5TokenizerFast.from_pretrained(model_path_input)
    model_input = T5ForConditionalGeneration.from_pretrained(model_path_input)

    # Extract question and correct answer
    correct_answer = input_text.get("answer", "")
    if not correct_answer:
        raise ValueError("correct_answer must be provided to validate distractors.")
    correct_answer_lower = correct_answer.lower()

    collected_distractors = set()
    raw_outputs = []

    for attempt in range(max_attempts):
        seed = random.randint(0, 1_000_000)
        torch.manual_seed(seed)

        preprocessed_input = preprocess_input(input_text)
        inputs = tokenizer_input(preprocessed_input, return_tensors="pt", padding=True)

        outputs = model_input.generate(
            **inputs,
            max_length=64,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.5,
            repetition_penalty=2.0,
            num_return_sequences=1
        )

        raw_output = tokenizer_input.decode(outputs[0], skip_special_tokens=True)
        raw_outputs.append(raw_output)

        # Clean and split
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
                return {
                    "input": preprocessed_input,
                    "output": str(collected_distractors),
                    "distractors": list(collected_distractors)
                }

        print(f"Attempt {attempt + 1} → Valid so far: {list(collected_distractors)}")

    print("⚠️ Max attempts reached. Returning what was collected.")
    return {
        "input": preprocessed_input,
        "output": raw_outputs,
        "distractors": list(collected_distractors)
    }