# batch_download_pdfs.py
from datasets import load_dataset
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import os

def convert_to_pdf(text, pdf_filename):
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    # Process line by line
    for line in text.split('\n'):
        if line.strip():
            clean_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(clean_line, styles['Normal']))
    doc.build(story)

print("Loading QASper dataset...")
dataset = load_dataset("allenai/qasper", split="validation", trust_remote_code=True)

NUM_PAPERS = 5 # We will start with 5 to keep evaluation time reasonable

print(f"Generating {NUM_PAPERS} PDFs...")
for i in range(NUM_PAPERS):
    item = dataset[i]
    title = item['title']
    print(f"[{i+1}/{NUM_PAPERS}] Processing: {title[:50]}...")
    
    # Extract text
    full_text = f"Title: {title}\n\nAbstract:\n{item['abstract']}\n\n"
    for section in item['full_text']['paragraphs']:
        for paragraph in section:
            full_text += paragraph + "\n"
    
    # Save directly as PDF
    pdf_name = f"qasper_paper_{i+1}.pdf"
    convert_to_pdf(full_text, pdf_name)

print("\n✅ All PDFs generated successfully!")