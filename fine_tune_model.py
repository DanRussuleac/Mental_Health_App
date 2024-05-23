import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Load the preprocessed dataset
data_dir = 'empatheticdialogues'

# Function to load the dataset
def load_empathetic_dialogues(data_dir, split):
    file_path = os.path.join(data_dir, f'{split}_reduced.csv')  # Use reduced dataset
    data = pd.read_csv(file_path, on_bad_lines='warn')
    return data

# Load the training data
train_data = load_empathetic_dialogues(data_dir, 'train')

# Convert the dialogues to a format suitable for the Hugging Face datasets library
train_dict = {
    "context": [],
    "response": [],
}

for i in range(len(train_data)):
    context = train_data.iloc[i]['context']
    response = train_data.iloc[i]['utterance']
    train_dict['context'].append(context)
    train_dict['response'].append(response)

# Create a Hugging Face dataset
train_dataset = Dataset.from_dict(train_dict)

def tokenize_function(examples):
    inputs = tokenizer(examples['context'], padding="max_length", truncation=True, max_length=512)
    outputs = tokenizer(examples['response'], padding="max_length", truncation=True, max_length=512)

    inputs['labels'] = outputs['input_ids']
    return inputs

tokenized_datasets = train_dataset.map(tokenize_function, batched=True)

# Prepare the dataset for training
train_dataset = tokenized_datasets.shuffle().select(range(len(train_dataset)))  # Use the full reduced dataset

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,  # Adjust batch size as needed
    per_device_eval_batch_size=2,
    num_train_epochs=1,  # Reduce number of epochs for quicker testing
    weight_decay=0.01,
)

trainer = Trainer(
    model=model.to(device),  # Move model to the device (GPU if available)
    args=training_args,
    train_dataset=train_dataset,
)

# Fine-tune the model
trainer.train()
model.save_pretrained("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")
