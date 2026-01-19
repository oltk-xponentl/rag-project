# **Enterprise Industry Analyst RAG**

This repository contains a **Retrieval-Augmented Generation (RAG)** application designed to analyze complex PDF reports across the Banking, Healthcare, and Manufacturing sectors.

This architecture implements advanced retrieval strategies-including **Hybrid Search**, **Reranking**, **Relevance Gating** and **Adaptive Diversity Filtering** to solve common production issues like "Context Crowding" and hallucination on irrelevant queries.

[View the App Here](https://unenslaved-radiogenic-tess.ngrok-free.dev/)

## **Key Features**

### **1\. Hybrid Retrieval (Ensemble)**

Combines **Dense Vector Search** (Semantic understanding via `Alibaba-NLP/gte-multilingual-base`) with **Sparse Keyword Search** (BM25). This ensures the system catches both high-level conceptual matches and specific acronyms/project names that vector models might miss.

### **2\. Multi-Stage Reranking Pipeline**

- **Recall Expansion:** Fetch 4x the needed documents (k=28) to create a wide funnel.
- **Flashrank Reranking:** A Cross-Encoder (`ms-marco-MiniLM-L-12-v2`) re-scores every candidate.
- **Adaptive Safety Valve:** A custom algorithm prevents one document from dominating the context window ("Context Crowding") while allowing high-confidence matches to bypass diversity caps (dynamic_threshold).

### **3\. Metadata Filtering & Attribution**

Users can limit the "Knowledge Scope" to specific industries. The system parses PDF metadata to cite specific source files and page numbers in the UI.

## **Tech Stack**

- **LLM Engine:** Ollama (`phi4-mini`)
- **Vector Store:** ChromaDB
- **Embeddings:** HuggingFace (`Alibaba-NLP/gte-multilingual-base`)
- **Reranker:** Flashrank (`ms-marco-MiniLM-L-12-v2`)
- **Orchestration:** LangChain (LCEL)
- **Frontend:** Streamlit

## **Project Structure**
```
rag-project  
├── app  
│ ┗ main.py # Streamlit Frontend  
├── data # PDF Source Documents  
│ ├── banking  
│ ├── healthcare  
│ ┗ manufacturing  
├── rag # Core Logic Package  
│ ├── ingestion.py # ETL Pipeline (Load -> Split -> Embed -> Chroma)  
│ ├── retriever.py # Hybrid Search & Smart Diversity Logic  
│ ┗ generator.py # LLM Chain & Relevance Gating  
├── chroma_db # Persistent Vector Database  
├── requirements.txt # Python Dependencies  
┗ README.md
```

## **Quick Start**

### **1\. Prerequisites**

- **Python 3.13+**
- **Ollama** installed and running.

### **2\. Install Dependencies**

```pip install -r requirements.txt```  

### **3\. Pull the Local LLM**

Ensure your Ollama instance has the required model:

```ollama pull phi4-mini```

### **4\. Ingest Data**

This step scans the data/ directory, splits the PDFs, generates embeddings, and saves them to chroma_db/.

```python -m rag.ingestion```

### **5\. Run the Application**

Launch the Streamlit interface:

```streamlit run app/main.py```  

## **Deep Dive: Architecture**

### **The Retrieval Pipeline (rag/retriever.py)**

The get_retriever function builds a `ContextualCompressionRetriever` with a custom `DebugContextualCompressionRetriever` wrapper for observability.

- **Initial Fetch (k\*4):** We request 28 documents from the Vector Store and BM25.
- **Ensemble:** These are weighted 50/50 to merge semantic and keyword relevance.
- **Rerank (ms-marco):** The top 21 candidates (k\*3) are passed to the Cross-Encoder. This model looks at the query and document _together_, providing a much more accurate score than vector distance.
- **Smart Diversity Filter (apply_smart_diversity):**
  - **The Problem:** Standard RAG often returns 7 chunks from the exact same page if that page is dense with keywords, losing context from other relevant reports.
  - **The Solution:** We enforce a max_per_doc limit (Default: 2).
  - **The Exception (Safety Valve):** If a chunk is exceptionally relevant (Score > Top Score - 0.02), it ignores the limit. This ensures we don't discard perfect answers just for the sake of diversity.

### **The Generator Logic (rag/generator.py)**

The generator module implements the final synthesis layer using LangChain Expression Language (LCEL) to orchestrate a deterministic and secure response pipeline.

- **Orchestration:** Uses `RunnableParallel` to execute retrieval and input passthrough in parallel. The retrieval step invokes the multi-stage retriever to gather the top 7 context chunks.
- **Context Synthesis:** Chunks are joined into a single string via the `format_docs` utility, which provides console-level debug information regarding source files and industry tags.
- **Relevance Gating:** The system prompt forces an immediate "Relevance Check." If the retrieved context contains no information, the model is strictly forbidden from hallucinating and must return a standard refusal.
- **Instruction Hierarchy & Security:** 
  - **XML Isolation:** User queries are encapsulated in `<user_query>` tags to distinguish between data and instructions.
  - **Anti-Injection:** Explicitly instruct the model to ignore any attempts to change its persona or instructions found within the user input.
- **Local Inference:** Utilizes `phi4-mini` with a temperature of 0.1 to ensure factual consistency and minimize variation across multiple runs.

## **Sample Questions for Testing**

Use the "Knowledge Scope" sidebar to filter by industry, or select "All" for cross-domain queries.

### **Banking (Select "Banking" in sidebar)**

"JPMorgan Chase identifies 'Agentic AI' as a top trend for 2025, but Deloitte predicts 2025 will be a 'gap year' for this technology. How do these two perspectives reconcile regarding the timeline for widespread adoption and the specific barriers preventing immediate scaling?"

- _Expectation:_ The system should synthesize the optimism from JPMC with the pragmatic "gap year" hurdles (legacy tech, governance) from Deloitte.

### **Sustainability & Tech (Select "All" or "Manufacturing")**

"Capgemini reports a slight slowdown in the growth of sustainability investments for 2025. How does this align with the rising energy demands of Generative AI data centers highlighted in the TMT Predictions?"

- _Expectation:_ This requires cross-referencing Capgemini's investment report with TMT's data center energy consumption stats.

### **Healthcare (Select "Healthcare")**

"How is the role of AI in pharmacovigilance evolving from 'rule-based' systems to 'AI agents'?"

- _Expectation:_ Should cite the specific _Intuition Labs_ or _FDA_ documents regarding signal detection and adverse event processing.

"What are the core ethical principles established by the WHO for the governance of AI in health?"

- _Expectation:_ Should list the specific principles (e.g., Protect autonomy, Promote human well-being, Ensure transparency) found in the WHO PDF.

### **Manufacturing (Select "Manufacturing")**

"What are the three primary domains where quantum technologies are expected to create value in advanced manufacturing?"

- _Expectation:_ Should pull from the WEF Quantum Tech report, likely citing optimization, simulation, and sensing.

"How are large organizations adjusting their supply chain strategies in 2025 regarding China?"

- _Expectation:_ Look for mentions of "China Plus One" strategies or diversification in the supply chain reports.

"UNCTAD reports a 5% decline in the value of greenfield industrial projects, yet IoT Analytics predicts a 13.5% CAGR boom in industrial software. What does this divergence suggest about the manufacturing investment strategy for 2025?"

- _Expectation:_ Should highlight a strategic shift in manufacturing from physical expansion to digital optimization, where a pullback in capital-intensive greenfield projects is offset by aggressive investment in software to drive efficiency

## **Guardrails**

The system prompt includes explicit instructions to:

- **Ignore Override Attempts:** If a user tries to jailbreak the persona ("Ignore previous instructions"), the model is instructed to refuse.
- **Fix PDF Artifacts:** Automatically corrects joined words (e.g., "growthin" -> "growth in") common in PDF extraction.
- **Numerical Precision:** Explicitly differentiates between Total Addressable Market (TAM) and Segment Serviceable Addressable Market (SAM).