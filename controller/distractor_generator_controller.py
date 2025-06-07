from transformers import T5TokenizerFast, T5ForConditionalGeneration

from helper import Helper

print("----- Initializing Distractor Generator Controller -----")
print("-- Loading Model and Tokenizer --")

model_path = "distractor_generator/model_final_base/final-model-2000"
tokenizer = T5TokenizerFast.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

print("-- Model and Tokenizer Loaded --")

def preprocess_input(input_text):
    """
    Preprocess the input text for the T5 model.
    """
    return f"Generate 3 distractors no duplicated for question: {input_text}"

def generate_distractors(input_text):
    preprocessed_input = preprocess_input(input_text)
    inputs = tokenizer(preprocessed_input, return_tensors="pt", padding=True)
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
    distractor_list = [d.strip() for d in raw_output.replace("Distractor:", "").split(",")]
    print(f"Input: {input_text}")
    print(f"Raw Output: {raw_output}")

    response = {
        "input": input_text,
        "output": raw_output,
        "distractors": distractor_list
    }

    return response

def generate_distractors_with_model(input_text, model_name):
    model_path_input = Helper.get_distractor_model_path(model_name)
    tokenizer_input = T5TokenizerFast.from_pretrained(model_path_input)
    model_input = T5ForConditionalGeneration.from_pretrained(model_path_input)

    preprocessed_input = preprocess_input(input_text)
    inputs = tokenizer_input(preprocessed_input, return_tensors="pt", padding=True)
    outputs = model_input.generate(
        **inputs,
        max_length=64,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=1.2,
        repetition_penalty=2.0,
        num_return_sequences=1
    )

    raw_output = tokenizer_input.decode(outputs[0], skip_special_tokens=True)
    distractor_list = [d.strip() for d in raw_output.replace("Distractor:", "").split(",")]
    print(f"Input: {input_text}")
    print(f"Raw Output: {raw_output}")

    response = {
        "input": input_text,
        "output": raw_output,
        "distractors": distractor_list
    }

    return response