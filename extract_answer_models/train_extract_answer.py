from transformers import T5TokenizerFast, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset, DatasetDict
import torch
import json
import random

# Load your dataset from JSON file
with open("extract_answer_models/dataset/cleaned_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

random.seed(42)
random.shuffle(data)

for i, d in enumerate(data[:10]):
    print(f"{i+1}. INPUT: {d['input']}")
    print(f"   OUTPUT: {d['output']}")

subset_size = 500
data_subset = data[:subset_size]

# Convert to Hugging Face Dataset
dataset = Dataset.from_list(data_subset)
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
# Load tokenizer and model
tokenizer = T5TokenizerFast.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Tokenization
def preprocess(example):
    model_input = tokenizer(example['input'], padding="max_length", truncation=True, max_length=64)
    target = tokenizer(example['output'], padding="max_length", truncation=True, max_length=64)
    model_input["labels"] = target["input_ids"]
    return model_input

# tokenized_dataset = dataset.map(preprocess)
tokenized_dataset = split_dataset.map(preprocess)
# Training arguments
print("Tokenization:", tokenizer.tokenize("_"))
print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("Training on subset size:", subset_size)
training_args = TrainingArguments(
    output_dir=f"./extract_answer_models/model_train/model-extract-{subset_size}",
    per_device_train_batch_size=4,
    num_train_epochs=10,
    learning_rate=3e-4,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=3,
    max_grad_norm=1.0,
    seed=42
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)


# Train the model
trainer.train()

final_model_dir = f"./extract_answer_models/model_final/final-model-{subset_size}"
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
print(f"✅ Final model saved to {final_model_dir}")