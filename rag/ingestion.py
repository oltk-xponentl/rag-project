import os
import shutil
import time
# Using PyMuPDFLoader as requested for stability
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "./chroma_db"
DATA_PATH = "./data"
BATCH_SIZE = 50  # Batch size for progress updates 

def ingest_data():
    """
    Ingests local PDFs into a ChromaDB vector store using batch processing.
    """
    if not os.path.exists(DATA_PATH):
        print(f"Directory {DATA_PATH} not found. Please create it and add industry subfolders.")
        return

    all_chunks = []
    
    print("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="Alibaba-NLP/gte-multilingual-base",
        model_kwargs={"trust_remote_code": True}
    )

    # --- Load and Chunk ---
    print(f"Scanning {DATA_PATH} for documents...")
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                industry_tag = os.path.basename(root)
                if industry_tag == "data":
                    industry_tag = "General"

                try:
                    loader = PyMuPDFLoader(file_path)
                    pages = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        add_start_index=True
                    )
                    
                    file_chunks = text_splitter.split_documents(pages)
                    
                    for chunk in file_chunks:
                        chunk.metadata["industry"] = industry_tag
                        chunk.metadata["source"] = file
                        if "page" in chunk.metadata:
                            chunk.metadata["page"] = chunk.metadata["page"] + 1
                    
                    all_chunks.extend(file_chunks)
                    print(f" Loaded: {file} | Industry: {industry_tag} | Chunks: {len(file_chunks)}")
                
                except Exception as e:
                    print(f" Error loading {file}: {str(e)}")

    if not all_chunks:
        print("No chunks were created.")
        return

    total_chunks = len(all_chunks)
    print(f"\nPreparing to embed {total_chunks} chunks...")

    # --- Initialize DB ---
    # Create the DB first with no data, just to set up the persistence folder
    if os.path.exists(CHROMA_PATH):
        print("Removing existing vector store...")
        shutil.rmtree(CHROMA_PATH)

    vector_store = Chroma(
        embedding_function=embedding_model,
        persist_directory=CHROMA_PATH
    )

    # --- Batched Ingestion ---
    print(f"Starting batch ingestion (Batch size: {BATCH_SIZE})...")
    start_time = time.time()

    for i in range(0, total_chunks, BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        
        # Add batch to vector store
        vector_store.add_documents(documents=batch)
        
        # Calculate progress
        current_count = min(i + BATCH_SIZE, total_chunks)
        elapsed = time.time() - start_time
        rate = current_count / elapsed if elapsed > 0 else 0
        remaining = (total_chunks - current_count) / rate if rate > 0 else 0
        
        print(f" [{current_count}/{total_chunks}] processed. Speed: {rate:.2f} chunks/sec. Est. remaining: {remaining/60:.1f} mins.")

    print(f"\n Success! Database saved to {CHROMA_PATH}")

if __name__ == "__main__":
    ingest_data()