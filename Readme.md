# **RAG System with FAISS and GPT**

This repository implements a **Retrieval-Augmented Generation (RAG)** system using **FAISS** for vector-based retrieval and **GPT** for generative response. It is designed to process large datasets, index them with FAISS, and use GPT to answer queries with retrieved context.

## **Features**

* **Document Loading** : Load and preprocess datasets (e.g., CSV, plain text).
* **Embedding Generation** : Convert documents and queries into vector embeddings.
* **Efficient Retrieval** : Use FAISS for similarity search over large corpora.
* **GPT Integration** : Generate answers using GPT with context from retrieved documents.
* **Modular Design** : Easily extend the system with new vector stores, LLMs, or document loaders.

## **Installation**

### **Requirements**

* Python 3.8 or higher
* FAISS
* OpenAI API Key

### **Dependencies**

Install the required packages using `pip`:

```bash
pip install -U -r requirements.txt
```

**`requirements.txt`:**

```
langchain
openai
faiss-cpu
python-dotenv
pandas
langchain-community
tiktoken
gradio
```

---

## **Usage**

### **1. Set Up Environment Variables**

Create a `.env` file in the project root directory and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key
```

### **2. Prepare Your Dataset**

* Place your documents in the `documents` directory.
* Supported formats: CSV, plain text, and others (easily extendable).

### **3. Run the System**

Execute the main script:

```bash
python main.py
```

### **4. Ask Questions**

Once the system is running, interact with it by typing your questions. For example:

```plaintext
Your question: What is the best way to handle data processing?

Answer: The best way to handle data processing is...
```

---

## **Project Structure**

```plaintext
.
├── documents/               # Directory for your dataset
├── rag_system/              # Main source code
│   ├── __init__.py
│   ├── config.py            # Configuration settings
│   ├── loaders.py           # Document loading and preprocessing
│   ├── vector_stores.py     # FAISS and other vector store implementations
│   ├── llms.py              # LLM integrations (e.g., OpenAI, HuggingFace)
│   ├── utils.py             # Helper utilities
│   └── main.py              # Entry point for the system
├── .env                     # Environment variables
├── requirements.txt         # Python dependencies
└── README.md                # Documentation (this file)
```

---

## **Customization**

### **1. Vector Store**

The system currently uses **FAISS** for vector-based retrieval. To use another vector database (e.g., Pinecone, Weaviate, Milvus):

1. Add a new class in `rag_system/vector_stores.py` inheriting from `BaseVectorStore`.
2. Update `Config.VECTOR_STORE_TYPE` in `config.py`.

### **2. LLM**

The system integrates with **OpenAI GPT** by default. To use another LLM (e.g., HuggingFace models):

1. Add a new class in `rag_system/llms.py` inheriting from `BaseLLM`.
2. Update `Config.LLM_TYPE` in `config.py`.

### **3. Document Loaders**

The system supports CSV and plain text files. To add support for other formats (e.g., PDFs, JSON):

1. Add a new class in `rag_system/loaders.py` inheriting from `BaseDocumentLoader`.
2. Modify the `load_documents` function to detect and handle the new format.

---

## **Future Improvements**

* **Scalable Vector Stores** : Add support for distributed databases like Pinecone or Weaviate.
* **Enhanced Query Handling** : Implement advanced query parsing and relevance ranking.
* **Fine-tuned Models** : Integrate domain-specific LLMs for improved performance.
* **Web Interface** : Build a frontend using Streamlit or Flask for user-friendly interaction.
* **Batch Processing** : Optimize retrieval and generation for bulk queries.

---

## **Contributing**

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with detailed explanations of your changes.

---

## **License**

This project is licensed under the [MIT License](https://chatgpt.com/c/LICENSE).

---

## **Acknowledgments**

* [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search.
* [OpenAI](https://openai.com/) for their powerful embedding and generative APIs.
* [LangChain](https://langchain.readthedocs.io/) for providing a robust framework for LLM applications.
