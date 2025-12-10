import os
import sys
from dotenv import load_dotenv

load_dotenv()

PDF_FILE_PATH = os.getenv('PDF_FILE_PATH', 'ai_engineer_cv.pdf')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'quickstart')
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'gemini').lower()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Validaciones críticas
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing via .env")

if LLM_PROVIDER == 'openai' and not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is required for openai provider")

if LLM_PROVIDER == 'gemini' and not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is required for gemini provider")