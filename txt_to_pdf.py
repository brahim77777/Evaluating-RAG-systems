# txt_to_pdf.py
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def convert_txt_to_pdf(txt_filename, pdf_filename):
    print(f"Converting {txt_filename} to {pdf_filename}...")
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    with open(txt_filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                # Clean characters for ReportLab
                clean_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(clean_line, styles['Normal']))
                
    doc.build(story)
    print("Done!")

if __name__ == "__main__":
    convert_txt_to_pdf("qasper_paper_1.txt", "qasper_paper_1.pdf")