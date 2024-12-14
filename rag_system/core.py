# rag_system/core.py

from langchain.chains import RetrievalQA
from rag_system.config import Config
from rag_system.loaders import load_documents
from rag_system.vector_stores import FAISSVectorStore
from rag_system.llms import OpenAILLM
from rag_system.utils import split_documents
from langchain.embeddings import OpenAIEmbeddings
import os

class RAGSystem:
    def __init__(self, proxy=None):
        if proxy:
            os.environ["http_proxy"] = proxy
            os.environ["https_proxy"] = proxy
        self.documents = []
        self.split_docs = []
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.qa_chain = None

    def load_and_prepare_documents(self):
        print("Loading documents...")
        self.documents = load_documents(
            Config.DOCUMENTS_DIRECTORY,
            Config.FILE_EXTENSION,
            Config.ENCODING
        )
        print(f"Loaded {len(self.documents)} documents.")
        
        print("Splitting documents into chunks...")
        self.split_docs = split_documents(
            self.documents,
            Config.CHUNK_SIZE,
            Config.CHUNK_OVERLAP
        )
        print(f"Created {len(self.split_docs)} document chunks.")

    def initialize_embeddings(self):
        print("Initializing OpenAI embeddings.")
        self.embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
        print("Initialized OpenAI embeddings.")

    def create_vector_store(self, index_type, num_clusters=100):
        print("Creating/loading vector store...")
        if index_type in ["Flat", "IVF", "HNSW", "IVFPQ", "HNSWSQ"]:
            self.vector_store = FAISSVectorStore(
                embeddings=self.embeddings,
                index_path=Config.VECTOR_STORE_INDEX_PATH,
                index_type=index_type,
                num_clusters=num_clusters
            )
        else:
            raise ValueError(f"Unsupported vector store type: {index_type}")
        
        self.vector_store.index_documents(self.split_docs)
        print("Vector store is ready.")

    def initialize_llm(self):
        print("Initializing LLM...")
        if Config.LLM_TYPE == "OpenAI":
            llm_instance = OpenAILLM(Config.OPENAI_API_KEY, Config.LLM_TEMPERATURE)
        else:
            raise ValueError(f"Unsupported LLM type: {Config.LLM_TYPE}")
        self.llm = llm_instance.get_llm()
        print("LLM is ready.")

    def build_qa_chain(self, chain_type):
        print("Building the RAG chain...")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=chain_type,
            retriever=self.vector_store.as_retriever(),
            return_source_documents=True
        )
        print("RAG chain is ready.")

    def initialize_system(self, index_type, num_clusters=100):
        self.load_and_prepare_documents()
        self.initialize_embeddings()
        self.create_vector_store(index_type, num_clusters)
        self.initialize_llm()
        self.build_qa_chain(Config.CHAIN_TYPE)

    def query(self, question):
        print("asking question")
        response = self.qa_chain.invoke(question)
        answer = response['result']
        print("show results: ", answer)
        source_docs = response['source_documents']
        sources = "\n".join([f"- {doc.metadata.get('source')} (Row {doc.metadata.get('row')})" for doc in source_docs[:3]])
        return answer, sources

    def add_documents(self, new_documents):
        # 假设 new_documents 是一个字典列表，每个字典代表一行数据
        self.documents.extend(new_documents)
        self.split_docs = split_documents(
            self.documents,
            Config.CHUNK_SIZE,
            Config.CHUNK_OVERLAP
        )
        self.vector_store.index_documents(self.split_docs)
        print(f"Added {len(new_documents)} new documents.")

    def delete_documents(self, criteria):
        """
        根据给定的条件删除文档。
        criteria: 字符串，例如 "id=123"
        """
        try:
            field, value = criteria.split('=')
            field = field.strip()
            value = value.strip()
            # 过滤文档
            self.documents = [doc for doc in self.documents if str(doc.get(field)) != value]
            # 重新拆分文档
            self.split_docs = split_documents(
                self.documents,
                Config.CHUNK_SIZE,
                Config.CHUNK_OVERLAP
            )
            # 重新索引向量数据库
            self.vector_store.index_documents(self.split_docs)
            print(f"Deleted documents where {field} = {value}.")
        except Exception as e:
            print(f"Error in deleting documents: {e}")

    def update_documents(self, criteria, new_values):
        """
        根据给定的条件更新文档。
        criteria: 字符串，例如 "id=123"
        new_values: 字符串，例如 "price=99.99"
        """
        try:
            crit_field, crit_value = criteria.split('=')
            crit_field = crit_field.strip()
            crit_value = crit_value.strip()

            update_field, update_value = new_values.split('=')
            update_field = update_field.strip()
            update_value = update_value.strip()

            for doc in self.documents:
                if str(doc.get(crit_field)) == crit_value:
                    doc[update_field] = update_value

            # 重新拆分文档
            self.split_docs = split_documents(
                self.documents,
                Config.CHUNK_SIZE,
                Config.CHUNK_OVERLAP
            )
            # 重新索引向量数据库
            self.vector_store.index_documents(self.split_docs)
            print(f"Updated documents where {crit_field} = {crit_value}. Set {update_field} = {update_value}.")
        except Exception as e:
            print(f"Error in updating documents: {e}")
