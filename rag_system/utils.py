# rag_system/utils.py

from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(documents: List[Dict], chunk_size: int, chunk_overlap: int) -> List[Dict]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs
