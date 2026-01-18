from typing import List, Dict, Any
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Configuration
CHROMA_PATH = "./chroma_db"

def apply_smart_diversity(reranked_results: List[Document], k: int = 7, max_per_doc: int = 2, hard_threshold_floor: float = 0.98, relative_drop: float = 0.02) -> List[Document]:
    """
    Filters results using Adaptive Safety Valve logic for high-confidence models.
    Logic:
    - Cap chunks per document at 'max_per_doc'.
    - Safety Valve: If a chunk's score is > max(hard_floor, top_score - drop), ignore cap.
    """
    final_context = []
    doc_counts = {}
    
    # Helper to safely extract score from various metadata formats
    def get_score(doc):
        return doc.metadata.get('relevance_score', 
               doc.metadata.get('score', 
               doc.metadata.get('cross_encoder_score', 0.0)))

    # Ensure results are sorted by score (descending) before filtering
    sorted_results = sorted(reranked_results, key=get_score, reverse=True)
    
    if not sorted_results:
        return []

    # Calculate Adaptive Threshold
    # Take the higher of the hard floor (0.98) OR the top score minus the drop (0.02)
    top_score = get_score(sorted_results[0])
    dynamic_threshold = max(hard_threshold_floor, top_score - relative_drop)

    for doc in sorted_results:
        # Stop if we filled the context window
        if len(final_context) >= k:
            break
            
        source = doc.metadata.get('source', 'Unknown')
        score = get_score(doc)
        
        # Track count per document
        current_count = doc_counts.get(source, 0)
        
        # Logic: Strict Cap with Adaptive Safety Valve
        # Accept if we are under the cap OR if the score beats the dynamic high bar
        if current_count < max_per_doc or score > dynamic_threshold:
            final_context.append(doc)
            doc_counts[source] = current_count + 1
        else:
            # Skip to force diversity
            pass 
            
    return final_context


class DebugContextualCompressionRetriever(ContextualCompressionRetriever):
    """
    Wrapper class that prints retrieval steps and applies Smart Diversity.
    """
    final_k: int = 7 # Target context size

    def __init__(self, final_k=7, **kwargs):
        super().__init__(**kwargs)
        self.final_k = final_k

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        # 1. Retrieve from Base (Ensemble: BM25 + Vector)
        docs = self.base_retriever.invoke(query)
        
        print(f"\n{'='*40}")
        print(f"[DEBUG] PHASE 1: Initial Hybrid Retrieval (Union of {len(docs)} docs)")
        print(f"{'-'*40}")
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', '?')
            snippet = doc.page_content[:50].replace('\n', ' ') + "..."
            print(f" Candidate {i+1:02d} | {source} (Pg {page}) | {snippet}")

        # 2. Rerank 
        if self.base_compressor:
            reranked_docs = self.base_compressor.compress_documents(docs, query)
        else:
            reranked_docs = docs

        print(f"\n{'-'*40}")
        print(f"[DEBUG] PHASE 2: Raw Rerank Scores (Top {len(reranked_docs)})")
        print(f"{'-'*40}")
        for i, doc in enumerate(reranked_docs):
            score = doc.metadata.get('relevance_score', 0.0)
            source = doc.metadata.get('source', 'Unknown')
            print(f" Rank {i+1:02d} | Score: {score:.4f} | {source}")

        # 3. Apply Smart Diversity (Filter Pool -> Final K)
        final_docs = apply_smart_diversity(
            reranked_docs, 
            k=self.final_k, 
            max_per_doc=4, 
            hard_threshold_floor=0.96,
            relative_drop=0.02
        )

        print(f"\n{'-'*40}")
        print(f"[DEBUG] PHASE 3: After Smart Diversity (Final {len(final_docs)})")
        print(f"{'-'*40}")
        
        for i, doc in enumerate(final_docs):
            # Using the same robust extraction for display
            score = doc.metadata.get('relevance_score', 0.0)
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', '?')
            print(f" Final {i+1:02d} | Score: {score:.4f} | {source} (Pg {page})")
        
        print(f"{'='*40}\n")
            
        return final_docs


def get_vectorstore():
    """
    Returns the initialized ChromaDB vector store.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name="Alibaba-NLP/gte-multilingual-base",
        model_kwargs={"trust_remote_code": True}
    )
    
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )

def get_retriever(industry_filter=None, k=7):
    """
    Builds the pipeline:
    1. Recall: Fetch 30+ candidates (BM25 + Vector)
    2. Rerank: Score top 20 candidates (Buffer Pool)
    3. Filter: Apply Smart Diversity to select final k (7)
    """
    vector_store = get_vectorstore()
    
    # Fetch 4x the target to feed the Reranker with a diverse pool
    initial_fetch_k = k * 4
    
    # Ask the Reranker for 2x or 3x the target to create a buffer
    rerank_pool_size = k * 3

    # Vector Retriever
    chroma_retriever = vector_store.as_retriever(
        search_kwargs={
            "k": initial_fetch_k, 
            "filter": {"industry": industry_filter} if industry_filter else None
        }
    )
    
    # BM25 Retriever
    where_filter = {"industry": industry_filter} if industry_filter else None
    try:
        all_data = vector_store.get(where=where_filter)
    except Exception:
        all_data = {"documents": [], "metadatas": []}

    docs = []
    if all_data['documents']:
        for i, text in enumerate(all_data['documents']):
            meta = all_data['metadatas'][i] if all_data['metadatas'] else {}
            docs.append(Document(page_content=text, metadata=meta))
    
    if not docs:
        print("Warning: No documents found for BM25 index. Defaulting to Vector Search.")
        base_retriever = chroma_retriever
    else:
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = initial_fetch_k

        base_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.5, 0.5]
        )

    # Reranker
    compressor = FlashrankRerank(
        model="ms-marco-MiniLM-L-12-v2", 
        top_n=rerank_pool_size
    )
    
    # --- Final Pipeline ---
    compression_retriever = DebugContextualCompressionRetriever(
        final_k=k,  
        base_compressor=compressor, 
        base_retriever=base_retriever
    )
    
    return compression_retriever