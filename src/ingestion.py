# import io
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Define the absolute path to the data folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "tp2")

def _resolve_path(file_path):
    # If an absolute path is provided, use it
    if os.path.isabs(file_path):
        return file_path
    
    # Join the absolute DATA_DIR with the filename
    full_path = os.path.join(DATA_DIR, file_path)
    
    print(f"Resolved path: {full_path}")
    return full_path

def load_pdf_content(file_path):
    path = _resolve_path(file_path)
    print(f"Loading PDF from: {path}")
    
    if not os.path.exists(path):
        # Debugging aid: print what is actually in the directory if file not found
        print(f"File not found. Contents of {DATA_DIR}:")
        try:
            print(os.listdir(DATA_DIR))
        except FileNotFoundError:
            print(f"Directory {DATA_DIR} does not exist.")
        raise FileNotFoundError(f"PDF not found: {path}")
        
    with open(path, 'rb') as f:
        pdf_reader = PdfReader(f)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    print(f"Extracted {len(text)} characters from the PDF.")
    return text

def process_text_to_docs(text, filename):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # For metadata, we resolve the path but only keep the basename to keep it clean
    source_path = _resolve_path(filename) if filename else ""
    docs = splitter.create_documents(
        texts=[text],
        metadatas=[{"source": os.path.basename(source_path) if source_path else filename}]
    )
    return docs