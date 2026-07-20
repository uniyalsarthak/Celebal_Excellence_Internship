
import os
from dotenv import load_dotenv

load_dotenv()  
# --- Chunking ---
CHUNK_SIZE = 500          
CHUNK_OVERLAP = 50        

# --- Embedding model (runs locally, free, no API key needed) ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Retrieval ---
TOP_K = 4                 

# --- Generation (Groq API — free tier available) ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_BASE = os.environ.get("GROQ_API_BASE", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
