import streamlit as st
import sys
import os

# Add the project root to the system path so imports work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.generator import generate_answer

# --- Page Config ---
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# --- Sidebar: Configuration & Filters ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Genpact_logo.svg/2560px-Genpact_logo.svg.png", width=200)
    st.title("Settings")
    
    # Feature: Metadata Filter
    st.markdown("### 🔍 Knowledge Scope")
    industry_choice = st.radio(
        "Select Industry Context:",
        ["All", "Banking", "Healthcare", "Manufacturing"],
        captions=["Search across all 30+ docs", "Finance, Risk & Insurance", "Life Sciences & Pharma", "Supply Chain & IoT"]
    )
    
    # Map the UI selection to ChromaDB metadata tags
    # These keys must match the folder names in /data
    industry_map = {
        "All": None,
        "Banking": "banking",
        "Healthcare": "healthcare", 
        "Manufacturing": "manufacturing"
    }
    
    selected_filter = industry_map[industry_choice]
    
    st.divider()
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Main Interface ---
st.title(" Industry Analyst AI")
st.markdown(f"**Current Focus:** `{industry_choice}`")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If there are sources associated with the assistant's message, show them
        if "sources" in message:
            with st.expander(" Referenced Sources"):
                for src in message["sources"]:
                    st.markdown(f"- **{src['source']}** (Page {src['page']})")

# Chat Input
if prompt := st.chat_input("Ask a question about Genpact's industries..."):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Answer
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents..."):
            try:
                # Call the RAG pipeline
                result = generate_answer(prompt, industry_filter=selected_filter)
                answer = result["answer"]
                sources = []
                
                # Extract citation data from source documents
                for doc in result["sources"]:
                    sources.append({
                        "source": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", "Unknown"),
                        "industry": doc.metadata.get("industry", "Unknown")
                    })
                
                st.markdown(answer)
                
                # Show citations immediately
                if sources:
                    with st.expander(" Referenced Sources"):
                        for src in sources:
                            st.markdown(f"- **{src['source']}** (Page {src['page']})")
                
                # 3. Save to History
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")