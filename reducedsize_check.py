import pandas as pd

reduced_data_path = 'empatheticdialogues/train_reduced.csv'
reduced_data = pd.read_csv(reduced_data_path, on_bad_lines='warn')
print(f"Number of rows in the reduced dataset: {len(reduced_data)}")
