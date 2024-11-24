# rag_system/loaders.py

import os
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict

class BaseDocumentLoader(ABC):
    @abstractmethod
    def load(self) -> List[Dict]:
        pass

class CSVLoader(BaseDocumentLoader):
    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path = file_path
        self.encoding = encoding

    def load(self) -> List[Dict]:
        try:
            df = pd.read_csv(self.file_path, encoding=self.encoding)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")

        documents = []
        for index, row in df.iterrows():
            row_text = ' | '.join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col]) and isinstance(row[col], str) ])
            documents.append({
                "page_content": row_text,
                "metadata": {"source": self.file_path, "row": index}
            })
        return documents

def load_documents(directory_path: str, file_extension: str, encoding: str) -> List[Dict]:
    documents = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                print(f"Loading file: {file_path}")
                loader = CSVLoader(file_path, encoding)  # Replace with appropriate loader
                docs = loader.load()
                documents.extend(docs)
    return documents
