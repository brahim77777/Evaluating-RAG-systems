# save_paper.py
from datasets import load_dataset

print("Loading dataset...")
dataset = load_dataset("allenai/qasper", split="validation", trust_remote_code=True)
first_item = dataset[0]

paper_title = first_item['title']
paper_abstract = first_item['abstract']
# The paper body is stored as a list of paragraphs under full_text
paragraphs = first_item['full_text']['paragraphs']

# Combine everything into one big string
full_paper_text = f"Title: {paper_title}\n\nAbstract:\n{paper_abstract}\n\n"
for section in paragraphs:
    for paragraph in section:
        full_paper_text += paragraph + "\n"

# Save it to a text file so your Rust RAG can read it
filename = "qasper_paper_1.txt"
with open(filename, "w", encoding="utf-8") as f:
    f.write(full_paper_text)

print(f"Successfully saved '{paper_title}' to {filename}!")
print(f"File size: {len(full_paper_text)} characters.")