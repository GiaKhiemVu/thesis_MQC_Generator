# 🧠 MQC Generator – Thesis Project Backend

This repository contains the backend system for the MQC (Multiple-choice Question) Generator, developed as part of a thesis project. It focuses on automating the creation of MCQs by extracting answers from source texts and generating plausible distractors.

## 🚀 Features

- **Answer Extraction**: Utilizes pre-trained models to identify key answers within input sentences.
- **Distractor Generation**: Generates plausible distractors to accompany the correct answer, enhancing the quality of MCQs.
- **Data Preprocessing**: Includes scripts to clean and prepare datasets for training and evaluation.
- **Evaluation Tools**: Provides tools to assess the performance of the extraction and generation models.

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**: Transformers (Hugging Face), PyTorch, JSON, and others
- **Models**: Pre-trained models for answer extraction and distractor generation

## 📁 Project Structure

```
├── distractor_generator/       # Modules for generating distractors
├── evaluation_data/            # Data and scripts for model evaluation
├── extract_answer_models/      # Models and scripts for answer extraction
├── original_dataset.json       # Original dataset used for training/testing
├── clean_original_data.py      # Script to clean and preprocess the dataset
├── test.py                     # Script to test the models
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore file
└── README.md                   # Project documentation
```

## 📦 Installation

1. **Clone the repository**:

```bash
git clone https://github.com/GiaKhiemVu/thesis_MQC_Generator.git
cd thesis_MQC_Generator
```

2. **Create a virtual environment** (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

## 🚴‍♂️ Usage

1. **Data Preprocessing**:

```bash
python clean_original_data.py
```

This script will clean the `original_dataset.json` and prepare it for model training or evaluation.

2. **Answer Extraction**:

Navigate to the `extract_answer_models/` directory and follow the instructions provided there to train or test the answer extraction models.

3. **Distractor Generation**:

Navigate to the `distractor_generator/` directory and follow the instructions provided there to generate distractors for the extracted answers.

4. **Testing**:

Use the `test.py` script to run tests on the models and view sample outputs.

## 🧪 Evaluation

The `evaluation_data/` directory contains datasets and scripts to evaluate the performance of the answer extraction and distractor generation models. Follow the instructions within that directory to conduct evaluations and view results.

## 📄 License

This project is licensed under the [MIT License](LICENSE).
