from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration
CHROMA_PATH = "./chroma_db"

def get_vectorstore():
    """
    Returns the initialized ChromaDB vector store.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name="Alibaba-NLP/gte-multilingual-base",
        model_kwargs={"trust_remote_code": True}
    )
    
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )
    return vector_store

def get_retriever(industry_filter=None, k=6):
    """
    Creates a retriever that fetches the top K most relevant chunks.
    
    Args:
        industry_filter (str, optional): If provided, restricts search to a specific industry.
        k (int): Number of chunks to retrieve.
    """
    vector_store = get_vectorstore()
    
    search_kwargs = {"k": k}
    
    # Apply Metadata Filter if an industry is selected
    if industry_filter and industry_filter != "All":
        search_kwargs["filter"] = {"industry": industry_filter}
    
    return vector_store.as_retriever(search_kwargs=search_kwargs)