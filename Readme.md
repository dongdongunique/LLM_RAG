# **RAG System with FAISS and GPT**

This repository implements a **Retrieval-Augmented Generation (RAG)** system using **FAISS** for vector-based retrieval and **GPT** for generative response. It is designed to process large datasets, index them with FAISS, and use GPT to answer queries with context retrieved from the documents.

## **Features**

- **Document Loading**: Load and preprocess datasets (e.g., CSV, plain text).
- **Embedding Generation**: Convert documents and queries into vector embeddings.
- **Efficient Retrieval**: Use FAISS for similarity search over large corpora.
- **GPT Integration**: Generate answers using GPT with context from retrieved documents.
- **Modular Design**: Easily extend the system with new vector stores, LLMs, or document loaders.
- **Interactive User Interface**: A Gradio-powered UI for easy interaction with the system. Supports:
  - Uploading and viewing CSV files.
  - Searching the indexed documents.
  - Managing document chunks.
  - Interacting with the system through intuitive input fields.
- **CRUD Operations**: Add, delete, update, and query document chunks in real-time.

---

## **Installation**

### **Requirements**

- Python 3.8 or higher
- FAISS
- OpenAI API Key
- Gradio (for the UI)

### **Dependencies**

Install the required packages using `pip`:

```bash
pip install -U -r requirements.txt
```

**`requirements.txt`:**

```plaintext
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

- Use the provided example dataset (`amazon_products.csv`) or upload your own CSV dataset.
- Ensure the dataset contains a column with text-based content (e.g., `description`) to generate embeddings.

### **3. Run the System**

You can run the system in two modes:

#### **Command-Line Interface (CLI)**

Run the system interactively via CLI:

```bash
python main.py
```

This mode allows you to interact with the RAG system by asking questions and retrieving answers.

#### **Gradio User Interface**

Run the system with the Gradio-powered UI:

```bash
python app.py
```

This launches a web-based interface for uploading datasets, managing document chunks, and querying the RAG system.

---

## **Project Structure**

```plaintext
.
├── amazon_products.csv       # Example dataset for testing
├── app.py                    # Gradio-based user interface
├── app.log                   # Log file for application events
├── dataset_cache.ipynb       # Notebook for dataset caching or analysis
├── main.py                   # CLI entry point for the RAG system
├── rag_system/               # Main source code for the RAG system
│   ├── __init__.py
│   ├── config.py             # Configuration settings (API keys, paths, etc.)
│   ├── core.py               # Core logic for RAG system initialization
│   ├── loaders.py            # Document loading and preprocessing
│   ├── llms.py               # Integration with GPT or other LLMs
│   ├── utils.py              # Utility functions (e.g., splitting documents)
│   ├── vector_stores.py      # FAISS and other vector store implementations
├── vector_store_index/       # Directory for storing FAISS index files
├── requirements.txt          # Python dependencies
├── SearchQ.md                # Markdown for documenting queries or use cases
├── README.md                 # Project documentation (this file)
├── .env                      # Environment variables for API keys
```

---

## **Gradio User Interface**

The Gradio UI provides an intuitive interface for interacting with the RAG system. It supports:

1. **Upload CSV Files**:
   - Upload datasets containing documents for indexing.
   - Automatically preprocesses and splits documents into chunks for embedding generation.

2. **Search Documents**:
   - Enter natural language queries in the search box.
   - The system retrieves the most relevant documents and generates a response using GPT.

3. **CRUD Operations**:
   - Add new document chunks to the indexed dataset.
   - Delete or update existing chunks based on specific criteria.

### **Starting the Gradio Interface**

Run the Gradio UI with:

```bash
python app.py
```

After launching, open the provided URL in a web browser to interact with the system.

---

## **Customization**

### **1. Vector Store**

The system uses **FAISS** for vector-based retrieval by default. To integrate another vector database (e.g., Pinecone, Weaviate, Milvus):

1. Create a new class in `rag_system/vector_stores.py` inheriting from `BaseVectorStore`.
2. Update the `Config.VECTOR_STORE_TYPE` in `config.py`.

### **2. LLM**

The system integrates with **OpenAI GPT**. To switch to another LLM (e.g., HuggingFace models):

1. Add a new class in `rag_system/llms.py` inheriting from `BaseLLM`.
2. Update `Config.LLM_TYPE` in `config.py`.

### **3. Document Loaders**

To support additional formats (e.g., PDFs, JSON):

1. Add a new class in `rag_system/loaders.py` inheriting from `BaseDocumentLoader`.
2. Update the `load_documents` function to detect and handle the new format.

### **4. Gradio UI**

The UI is modular and can be extended. To add new components:

1. Modify `app.py` to include new Gradio widgets.
2. Update the callback functions to handle the added functionality.

---

## **Logging**

The system logs important events (e.g., errors, indexing operations) in `app.log`. Check this file for debugging or monitoring purposes.

---

## **Future Improvements**

- **Distributed Vector Stores**: Add support for scalable vector stores like Pinecone or Weaviate.
- **Advanced Query Features**: Implement query expansion, semantic search, and ranking.
- **Custom Embeddings**: Allow users to upload precomputed embeddings.
- **User Authentication**: Add authentication and access control for the Gradio interface.
- **Visualization**: Display results with data visualizations (e.g., charts for document relevance scores).
- **Batch Processing**: Optimize retrieval and generation for bulk queries.

---

## **Contributing**

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with detailed explanations of your changes.

---

## **License**

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## **Acknowledgments**

- [FAISS](https://github.com/facebookresearch/faiss): Efficient similarity search.
- [OpenAI](https://openai.com/): Embedding and generative APIs.
- [LangChain](https://langchain.readthedocs.io/): Framework for building applications with LLMs.
- [Gradio](https://gradio.app/): User-friendly interface for ML applications.
