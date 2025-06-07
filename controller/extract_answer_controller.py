import os
from transformers import T5TokenizerFast, T5ForConditionalGeneration

from helper import Helper

print("----- Initializing Extract Answer Controller -----")
print("-- Loading Model and Tokenizer --")

model_path = os.path.join("extract_answer_models", "model_final", "final-model-2000")
tokenizer = T5TokenizerFast.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

print("-- Model and Tokenizer Loaded --")

def preprocess_input(input_text):
    """
    Preprocess the input text for the T5 model.
    """
    return f"Convert this sentence into fill in blank: {input_text}"

def generate_answer_with_model(input_text, model_name):
    model_path = Helper.get_extract_model_path(model_name)
    tokenizer = T5TokenizerFast.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    """
    Generate an answer using the T5 model.
    """
    preprocessed_input = preprocess_input(input_text)
    inputs = tokenizer(preprocessed_input, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs, max_length=64)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Input: {input_text}")
    print(f"Output: {decoded_output}")

    response = {
        "input": input_text,
        "output": decoded_output
    }
    return response

def generate_answer(input_text):
    """
    Generate an answer using the T5 model.
    """
    preprocessed_input = preprocess_input(input_text)
    inputs = tokenizer(preprocessed_input, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs, max_length=64)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Input: {input_text}")
    print(f"Output: {decoded_output}")

    response = {
        "input": input_text,
        "output": decoded_output
    }
    return response