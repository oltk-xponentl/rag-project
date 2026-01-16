from operator import itemgetter
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from rag.retriever import get_retriever

# Strict prompt for Genpact Project Requirements
PROMPT_TEMPLATE = """
You are an expert Enterprise Analyst. Your task is to generate a comprehensive answer based on the provided context.

<context>
{context}
</context>

### Guidelines for Response:

1. **Relevance Check:**
   - If the context has NO relevant information, reply: "I'm sorry, but this information is not available in the provided documents."
   - If it has relevant info, proceed immediately.

2. **Strict Formatting Rules:**
   - **Fix Spacing:** The context text may have joined words (e.g., "$1.9trillion", "growthin", "2025report"). You MUST insert correct spaces in your output (e.g., "$1.9 trillion", "growth in", "2025 report").
   - **Plain Text Only:** Do NOT use italics or bold text for emphasis within paragraphs. Write in standard, plain English.
   - **Numbers:** Ensure there is always a space between a number and a word (e.g., write "30 %" or "30 percent", not "30%").

3. **Detail & Structure:**
   - Provide a detailed, multi-paragraph explanation.
   - Expand on the metrics and trends found in the text.
   - Do not summarize; be exhaustive and thorough.

4. **No Meta-Commentary or Tags:**
   - Do not explain that you fixed the text.
   - Do NOT use citation tags like `[Source: Name]` or `[Context]`.
   - Just provide the clean, corrected answer.

### Question:
{input}

### Answer:
"""

def format_docs(docs):
    """
    Joins retrieved documents into a single string and prints debug info to console.
    """
    if not docs:
        print("DEBUG: No documents were retrieved!")
        return ""
    
    print(f"DEBUG: Retrieved {len(docs)} chunks.")
    for i, doc in enumerate(docs):
        print(f"DEBUG: Chunk {i+1} from {doc.metadata.get('source')} (Industry: {doc.metadata.get('industry')})")
    
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(industry_filter=None):
    # Initialize LLM
    llm = ChatOllama(model="phi4-mini", temperature=0.1)
    
    # Normalize industry filter to lowercase to match folder-based ingestion
    normalized_filter = industry_filter.lower() if industry_filter else None
    
    # Get Retriever
    retriever = get_retriever(industry_filter=normalized_filter, k=7)
    
    # Create the Prompt
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # Construct the LCEL Chain
    retrieval_step = RunnableParallel({
        "context": itemgetter("input") | retriever,
        "input": itemgetter("input")
    })

    rag_chain = retrieval_step.assign(
        answer=(
            RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
            | prompt
            | llm
            | StrOutputParser()
        )
    )
    
    return rag_chain

def generate_answer(question: str, industry_filter: str = None):
    chain = get_rag_chain(industry_filter)
    response = chain.invoke({"input": question})
    
    return {
        "answer": response["answer"],
        "sources": response["context"]
    }