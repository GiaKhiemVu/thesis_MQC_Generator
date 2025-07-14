from flask import Flask, request, jsonify
from controller.distractor_generator_controller import generate_distractors, preprocess_input as preprocess_distractor_input, generate_distractors_with_model
from controller.extract_answer_controller import generate_answer, preprocess_input as preprocess_answer_input, generate_answer_with_model
from helper import Helper
from flask_cors import CORS

print("----- Initializing Flask App -----")
app = Flask(__name__)

print("----- Enabling CORS -----")
print("CORS is enabled for http://localhost:3000")

CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

print("CORS enabled successfully")

@app.route('/create-multiple-choices-details', methods=['POST'])
def create_multiple_choices_details():
    print(request.json)
    data = request.json
    input_text = data.get('input_text', '')

    extract_model_name = data.get('extract_model')
    distractor_model_name = data.get('distractor_model')
    
    if not input_text or not extract_model_name or not distractor_model_name:
        return jsonify({"error": "Required fields is missing"}), 400
    
    input_to_model1 = preprocess_answer_input(input_text)
    extract_result = generate_answer_with_model(input_text, extract_model_name)

    step1_convert_to_step2 = Helper.convert_output_step_1_into_input_step_2(extract_result['output'])

    input_to_model2 = preprocess_distractor_input(step1_convert_to_step2)
    distractor_result = generate_distractors_with_model(step1_convert_to_step2, distractor_model_name)

    response = {
        "input_text": input_text,
        "input_to_model1": input_to_model1,
        "extracted_result": extract_result,
        "converted_input_step_2": f"Question: {step1_convert_to_step2['question']}, Answer: {step1_convert_to_step2['answer']}",
        "input_to_model2": input_to_model2,
        "distractor_result": distractor_result
    }

    return jsonify(response)

@app.route('/create-multiple-choices', methods=['POST'])
def create_multiple_choices():
    data = request.json
    input_text = data.get('input_text', '')
    
    if not input_text:
        return jsonify({"error": "Input text is required"}), 400
    
    extract_result = generate_answer(input_text)
    step2_input = Helper.convert_output_step_1_into_input_step_2(extract_result['output'])
    distractor_result = generate_distractors(step2_input)

    response = Helper.extract_question_and_answer(extract_result['output'])
    response['distractors'] = distractor_result['distractors']
    print(response)
    return jsonify(response)

@app.route('/extract-answer', methods=['POST'])
def extract_answer():
    data = request.json
    input_text = data.get('input_text', '')
    
    if not input_text:
        return jsonify({"error": "Input text is required"}), 400
    
    extract_answer = generate_answer(input_text)

    return jsonify(extract_answer)

@app.route('/distractors-generator', methods=['POST'])
def istractors_generator():
    data = request.json
    input_text = data.get('input_text', '')
    
    if not input_text:
        return jsonify({"error": "Input text is required"}), 400
    
    distracts_result = generate_distractors(input_text)

    return jsonify(distracts_result)

if __name__ == '__main__':
    app.run(debug=True)