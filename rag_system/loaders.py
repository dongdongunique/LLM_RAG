# rag_system/loaders.py

import os
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict
# import datasets
class BaseDocumentLoader(ABC):
    @abstractmethod
    def load(self) -> List[Dict]:
        pass

class Document:
    def __init__(self, page_content: str, metadata: Dict):
        self.page_content = page_content
        self.metadata = metadata

class CSVLoader(BaseDocumentLoader):
    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path = file_path
        self.encoding = encoding

    def load(self) -> List[Document]:
        try:
            df = pd.read_csv(self.file_path, encoding=self.encoding)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")

        documents = []
        for index, row in df.iterrows():
            row_text = ' | '.join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
            document = Document(
                page_content=row_text,
                metadata={"source": self.file_path, "row": index}
            )
            documents.append(document)
        return documents
# class AmazonProductLoader(BaseDocumentLoader):
#     def load(self) -> List[Dict]:
#         try:
#             df = 
#         except Exception as e:
#             raise ValueError(f"Error reading CSV file: {e}")

#         documents = []
#         for index, row in df.iterrows():
#             row_text = ' | '.join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col]) and isinstance(row[col], str)])
#             document = Document(
#                 page_content=row_text,
#                 metadata={"source": self.file_path, "row": index}
#             )
#             documents.append(document)
#         return documents
def load_documents(directory_path: str, file_extension: str, encoding: str) -> List[Document]:
    documents = []
    #if directory_path ends with file_extension, then load the file
    if directory_path.endswith(file_extension):
        print(f"Loading file: {directory_path}")
        loader = CSVLoader(directory_path, encoding)  # Replace with appropriate loader
        docs = loader.load()
        documents.extend(docs)
        return documents
    else:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith(file_extension):
                    file_path = os.path.join(root, file)
                    print(f"Loading file: {file_path}")
                    loader = CSVLoader(file_path, encoding)  # Replace with appropriate loader
                    docs = loader.load()
                    documents.extend(docs)
        return documents
