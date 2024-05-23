import os
import pandas as pd

data_dir = 'empatheticdialogues'

# Function to load and reduce the dataset
def reduce_dataset(data_dir, split, fraction=0.2):
    file_path = os.path.join(data_dir, f'{split}.csv')
    data = pd.read_csv(file_path, on_bad_lines='warn')
    reduced_data = data.sample(frac=fraction, random_state=1)
    reduced_file_path = os.path.join(data_dir, f'{split}_reduced.csv')
    reduced_data.to_csv(reduced_file_path, index=False)
    return reduced_file_path

# Reduce the training data
train_file_path = reduce_dataset(data_dir, 'train', fraction=0.1)  # Adjust fraction to 0.1 for 10%
print(f"Reduced dataset saved to: {train_file_path}")
