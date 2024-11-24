# rag_system/config.py

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI API Key
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Document settings
    DOCUMENTS_DIRECTORY = "documents"
    FILE_EXTENSION = ".csv"
    ENCODING = "utf-8"

    # Vector store settings
    VECTOR_STORE_TYPE = "FAISS"  # Options: FAISS, Pinecone, Weaviate, etc.
    VECTOR_STORE_INDEX_PATH = "vector_store_index"

    # LLM settings
    LLM_TYPE = "OpenAI"  # Options: OpenAI, HuggingFace, etc.
    LLM_TEMPERATURE = 0.7

    # Text splitter settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # Chain settings
    CHAIN_TYPE = "stuff"  # Options: stuff, map_reduce, refine
