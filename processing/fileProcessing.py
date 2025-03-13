import docx
import pdfplumber
import pandas as pd


# Function to extract text from CSV
def extract_text_from_csv(csv_file):
    df = pd.read_csv(csv_file, converters={"resume_text": lambda x: x.strip()})
    return " ".join(df["resume_text"].tolist())  # Combine all text into a single string


# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()


# Function to extract text from DOCX
def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()
