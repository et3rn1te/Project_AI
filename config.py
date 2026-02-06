"""
Configuration settings for RAG Learning System
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ollama Settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b")

# Embedding Model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ChromaDB Settings
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

# Document Processing Settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Retrieval Settings
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

# LLM Generation Settings
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500"))

# Data Directories
DATA_DIR = "./data"
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
MARKDOWN_DIR = os.path.join(DATA_DIR, "markdown")
TEXT_DIR = os.path.join(DATA_DIR, "text")
