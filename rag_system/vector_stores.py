# rag_system/vector_stores.py

import os
from abc import ABC, abstractmethod
from typing import List, Dict
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS

class BaseVectorStore(ABC):
    @abstractmethod
    def index_documents(self, documents: List[Dict]):
        pass

    @abstractmethod
    def as_retriever(self):
        pass

class FAISSVectorStore(BaseVectorStore):
    def __init__(self, embeddings: Embeddings, index_path: str):
        self.embeddings = embeddings
        self.index_path = index_path
        self.vector_store = None

    def index_documents(self, documents: List[Dict]):
        if os.path.exists(self.index_path + ".index"):
            self.vector_store = FAISS.load_local(self.index_path, self.embeddings)
            print("Loaded existing FAISS index.")
        else:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            self.vector_store.save_local(self.index_path)
            print("Created and saved new FAISS index.")

    def as_retriever(self):
        return self.vector_store.as_retriever()
