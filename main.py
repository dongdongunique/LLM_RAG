# rag_system/main.py

from langchain.chains import RetrievalQA
from rag_system.config import Config
from rag_system.loaders import load_documents
from rag_system.vector_stores import FAISSVectorStore  # Adjust when adding new vector stores
from rag_system.llms import OpenAILLM  # Adjust when adding new LLMs
from rag_system.utils import split_documents

def main():
    # Load documents
    print("Loading documents...")
    documents = load_documents(
        Config.DOCUMENTS_DIRECTORY,
        Config.FILE_EXTENSION,
        Config.ENCODING
    )
    print(f"Loaded {len(documents)} documents.")
    #print(f"Example document: {documents['page_contents']}")
    # Split documents into chunks
    print("Splitting documents into chunks...")
    split_docs = split_documents(
        documents,
        Config.CHUNK_SIZE,
        Config.CHUNK_OVERLAP
    )
    print(f"Created {len(split_docs)} document chunks.")

    # Initialize embeddings
    from langchain.embeddings import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
    print("Initialized OpenAI embeddings.")
    #show the embeddings' shape
    #print(f"Embeddings shape: {embeddings.embeddings.shape}")
    # Create vector store
    print("Creating/loading vector store...")
    if Config.VECTOR_STORE_TYPE == "FAISS":
        vector_store = FAISSVectorStore(embeddings, Config.VECTOR_STORE_INDEX_PATH)
    else:
        raise ValueError(f"Unsupported vector store type: {Config.VECTOR_STORE_TYPE}")

    vector_store.index_documents(split_docs)

    # Initialize LLM
    print("Initializing LLM...")
    if Config.LLM_TYPE == "OpenAI":
        llm_instance = OpenAILLM(Config.OPENAI_API_KEY, Config.LLM_TEMPERATURE)
    else:
        raise ValueError(f"Unsupported LLM type: {Config.LLM_TYPE}")

    llm = llm_instance.get_llm()

    # Build the RAG chain
    print("Building the RAG chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=Config.CHAIN_TYPE,
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    print("RAG chain is ready.")

    # Interactive Q&A
    print("\nYou can now ask questions! Type 'exit' to quit.")
    while True:
        query = input("\nYour question: ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting the Q&A session.")
            break
        response = qa_chain.invoke(query)
        answer = response['result']
        source_docs = response['source_documents']

        print(f"\nAnswer: {answer}")
        print("\nSource Documents:")
        for doc in source_docs[:3]:
            print(f"- {doc.metadata.get('source')} (Row {doc.metadata.get('row')})")

if __name__ == "__main__":
    main()
