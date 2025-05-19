thesis_MQC_Generator
A Python-based system for generating multiple-choice questions (MCQs) from textual datasets, developed as part of a thesis project.

📘 Overview
This project focuses on automating the creation of MCQs by extracting answers from source texts and generating plausible distractors. It aims to assist educators and content creators in efficiently producing assessment materials.

📂 Project Structure
extract_answer_models/: Contains models and scripts for extracting potential answers from the dataset.

distractor_generator/: Includes modules responsible for generating distractor options for each question.

evaluation_data/: Stores data used for evaluating the quality and effectiveness of the generated MCQs.

original_dataset.json: The primary dataset used as the source for question generation.

clean_original_data.py: Script to preprocess and clean the original dataset.

test.py: Script to test the functionalities of the MCQ generation pipeline.

.gitignore: Specifies files and directories to be ignored by Git.

README.md: Provides an overview and instructions for the project.

🚀 Getting Started
Prerequisites
Python 3.7 or higher

Recommended: Create a virtual environment to manage dependencies.

Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/GiaKhiemVu/thesis_MQC_Generator.git
cd thesis_MQC_Generator
Install required dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Note: Ensure that a requirements.txt file is present. If not, you'll need to manually install the necessary packages.

🛠 Usage
Preprocess the dataset:

bash
Copy
Edit
python clean_original_data.py
This will clean and prepare the dataset for MCQ generation.

Generate MCQs:

bash
Copy
Edit
python test.py
This script will execute the MCQ generation pipeline, producing questions and corresponding distractors.

📊 Evaluation
The evaluation_data/ directory contains tools and datasets to assess the quality of the generated MCQs. This includes metrics for evaluating the relevance and plausibility of distractors, as well as the overall coherence of the questions.

🤝 Contributing
Contributions are welcome! If you'd like to enhance the functionality, fix bugs, or add new features, please fork the repository and submit a pull request.

📄 License
This project is open-source and available under the MIT License.

📬 Contact
For questions, suggestions, or collaborations, please contact Gia Khiem Vu.

Feel free to customize this README.md further to match the specific details and requirements of your project.