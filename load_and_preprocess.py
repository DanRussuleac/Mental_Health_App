import os
import pandas as pd
from datasets import Dataset

data_dir = 'empatheticdialogues'

# Function to load the dataset
def load_empathetic_dialogues(data_dir, split):
    file_path = os.path.join(data_dir, f'{split}.csv')
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
    response = train_data.iloc[i]['utterance']  # Use 'utterance' for responses
    train_dict['context'].append(context)
    train_dict['response'].append(response)

# Create a Hugging Face dataset
train_dataset = Dataset.from_dict(train_dict)

print("Sample from the dataset:", train_dataset[0])
