# rag_system/vector_stores.py

import os
from abc import ABC, abstractmethod
from typing import List, Dict
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
import faiss  # Import FAISS library directly for advanced index support
import numpy as np
from langchain.docstore.in_memory import InMemoryDocstore
    
class BaseVectorStore(ABC):
    @abstractmethod
    def index_documents(self, documents: List[Dict]):
        pass

    @abstractmethod
    def as_retriever(self):
        pass

class FAISSVectorStore(BaseVectorStore):
    def __init__(self, embeddings: Embeddings, index_path: str, index_type: str = "Flat", num_clusters: int = 100, ef_construction: int = 200, m: int = 16):
        """
        Initialize FAISS Vector Store.

        Args:
            embeddings: Embedding model to generate vector representations.
            index_path: Path to save/load FAISS index.
            index_type: Type of FAISS index ('Flat', 'IVF', or 'HNSW').
            num_clusters: Number of clusters for IVF index (ignored for 'Flat' and 'HNSW').
            ef_construction: Parameter for the construction of the HNSW index (higher values lead to more accuracy).
            m: Number of connections for each node in the HNSW graph (higher values improve recall).
        """
        self.embeddings = embeddings
        self.index_path = index_path
        self.index_type = index_type
        self.num_clusters = num_clusters
        self.ef_construction = ef_construction
        self.m = m
        self.n_bits = 8
        self.vector_store = None    

    def _create_flat_index(self, dimension: int):
        return faiss.IndexFlatL2(dimension)

    def _create_ivf_index(self, dimension: int):
        quantizer = self._create_flat_index(dimension)  # Flat index as the quantizer
        index = faiss.IndexIVFFlat(quantizer, dimension, self.num_clusters, faiss.METRIC_L2)
        return index
    
    def _create_ivfpq_index(self, dimension: int):
        quantizer = self._create_flat_index(dimension)  # Flat index as the quantizer
        index = faiss.IndexIVFPQ(quantizer, dimension, self.num_clusters, self.m, self.n_bits)
        return index
    
    def _create_hnsw_index(self, dimension: int):
        index = faiss.IndexHNSWFlat(dimension, self.m)  # Create the HNSW index
        index.hnsw.efConstruction = self.ef_construction  # Set the ef_construction parameter
        return index
    def _create_hnsw_sq_index(self, dimension: int):
        hnsw_index = faiss.IndexHNSWFlat(dimension, self.m)
        hnsw_index.hnsw.efConstruction = self.ef_construction
        
        quantizer = faiss.IndexFlatL2(dimension)  # Quantizer for PQ (flat L2 distance)
        pq_index = faiss.IndexIVFPQ(quantizer, dimension, self.num_clusters, self.n_bits, faiss.METRIC_L2)
        
        return hnsw_index, pq_index

    def _train_index(self, index, embeddings):
        """Train the index if it requires training (e.g., IVF)."""
        if index.is_trained:
            print("Index is already trained.")
        else:
            print("Training the FAISS index...")
            index.train(embeddings)

    def index_documents(self, documents: List[Dict]):
        """
        Index documents into the FAISS vector store.

        Args:
            documents: List of dictionaries containing document data.
        """
        # Check if the index already exists
        if os.path.exists(self.index_path + ".index"):
            self.vector_store = FAISS.load_local(self.index_path, self.embeddings)
            print("Loaded existing FAISS index.")
            return

        # Generate embeddings
        print("Generating embeddings for documents...")
        embeddings = self.embeddings.embed_documents([doc.page_content for doc in documents])
        
        embeddings = np.array(embeddings)
        dimension = embeddings.shape[1]

        # Create the appropriate FAISS index
        if self.index_type == "Flat":
            print("Creating a Flat index...")
            index = self._create_flat_index(dimension)
        elif self.index_type == "IVF":
            print(f"Creating an IVF index with {self.num_clusters} clusters...")
            index = self._create_ivf_index(dimension)
            self._train_index(index, embeddings)
        elif self.index_type == "IVFPQ":
            print(f"Creating an IVF-PQ index with {self.num_clusters} clusters and {self.m} subquantizers...")
            index = self._create_ivfpq_index(dimension)
            self._train_index(index, embeddings)
        elif self.index_type == "HNSW":
            print(f"Creating an HNSW index with m={self.m} and ef_construction={self.ef_construction}...")
            index = self._create_hnsw_index(dimension)
        elif self.index_type == "HNSWSQ":
            print(f"Creating an HNSW + SQ index with m={self.m}, ef_construction={self.ef_construction}, and n_bits={self.n_bits}...")
            hnsw_index, pq_index = self._create_hnsw_sq_index(dimension)
            # Train the indices separately
            self._train_index(hnsw_index, embeddings)
            self._train_index(pq_index, embeddings)
            # Add embeddings to the indices
            hnsw_index.add(embeddings)
            pq_index.add(embeddings)
            # Use both indices for search (not a single combined index)
            self.vector_store = FAISS(
                index=hnsw_index,  # Use HNSW index for searching
                embedding_function=self.embeddings.embed_query,
                docstore=InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)}),
                index_to_docstore_id={i: str(i) for i in range(len(documents))}
            )
            self.vector_store.save_local(self.index_path)
            return
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # Add embeddings to the index
        print("Adding embeddings to the index...")
        index.add(embeddings)

        # Wrap the FAISS index with LangChain's FAISS vector store for compatibility
        # self.vector_store = FAISS(index, self.embeddings)
        self.vector_store = FAISS(
            index=index,
            embedding_function=self.embeddings.embed_query,
            docstore=InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)}),
            index_to_docstore_id={i: str(i) for i in range(len(documents))}
        )
        
        self.vector_store.save_local(self.index_path)
        print(f"Created and saved new {self.index_type} FAISS index.")

    def as_retriever(self):
        """Convert the vector store to a retriever for querying."""
        return self.vector_store.as_retriever()
