print("----- Initializing Helper Class -----")

class Helper:
    @staticmethod
    def convert_output_step_1_into_input_step_2(output):
        if type(output) is not str:
            print("Output is not a string, returning as is.")
            return output
        print(f"Converting output from step 1 to input for step 2... {output}")
        if "|||" in output:
            question_part, answer_part = output.split("|||")
            question = question_part.replace("question:", "").strip()
            answer = answer_part.replace("answer:", "").strip()
            return {
                "question": question,
                "answer": answer
            }
        return output.strip()
    
    @staticmethod
    def extract_question_and_answer(input_str):
        if "Output:" in input_str:
            input_str = input_str.replace("Output:", "").strip()
        if "|||" in input_str:
            question_part, answer_part = input_str.split("|||")
            question = question_part.replace("question:", "").strip()
            answer = answer_part.replace("answer:", "").strip()
            return {
                "question": question,
                "answer": answer
            }
        return {}
    
    @staticmethod
    def get_extract_model_path(model_name):
        model_paths = {
            "t5-small-500": "extract_answer_models/model_final/final-model-500",
            "t5-small-1000": "extract_answer_models/model_final/final-model-1000",
            "t5-small-1500": "extract_answer_models/model_final/final-model-1500",
            "t5-small-2000": "extract_answer_models/model_final/final-model-2000",
            "t5-small-2999": "extract_answer_models/model_final/final-model-2999",
        }
        return model_paths.get(model_name, "")
    
    @staticmethod
    def get_distractor_model_path(model_name):
        model_paths = {
            "t5-small-500": "distractor_generator/model_final/final-model-500",
            "t5-small-1000": "distractor_generator/model_final/final-model-1000",
            "t5-small-1500": "distractor_generator/model_final/final-model-1500",
            "t5-small-2000": "distractor_generator/model_final/final-model-2000",
            "t5-base-2000": "distractor_generator/model_final_base/final-model-2000",
            "t5-base-2999": "distractor_generator/model_final_base/final-model-2999",
        }
        return model_paths.get(model_name, "")
print("-- Helper Class Initialized --")