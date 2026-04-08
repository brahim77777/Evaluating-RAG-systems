# prepare_eval.py
from datasets import load_dataset

print("Downloading QASper dataset (this might take a minute)...")
# We load the 'validation' split because it's smaller and perfect for testing
dataset = load_dataset("allenai/qasper", split="validation")

# Let's inspect the very first paper in the dataset
first_item = dataset[0]

print("\n--- Dataset Successfully Loaded ---")
print(f"Paper Title: {first_item['title']}")
print(f"Number of questions for this paper: {len(first_item['qas']['question'])}")
print(f"First Question: {first_item['qas']['question'][0]}")

# We will look at the answers next!