import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import streamlit as st
import pandas as pd
import torch
from rag_api import RAGEngine

# Set Streamlit page config
st.set_page_config(
    page_title="RBI Insights | Local RAG Compliance Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling (Dark Mode & Fintech Theme)
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Titles and text styling */
    h1, h2, h3 {
        color: #38bdf8 !important;
        font-family: 'Inter', sans-serif;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Metrics panel */
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .metric-val {
        font-size: 1.8rem;
        font-weight: bold;
        color: #38bdf8;
    }
    
    .metric-lbl {
        color: #64748b;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Chat message bubble */
    .chat-bubble {
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        line-height: 1.6;
        color: #f1f5f9 !important;
    }
    
    .chat-user {
        background-color: #1e293b;
        border-left: 4px solid #3b82f6;
    }
    
    .chat-assistant {
        background-color: rgba(15, 23, 42, 0.8);
        border-left: 4px solid #10b981;
        border: 1px solid #1e293b;
    }
    
    /* Citation block */
    .citation-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 0.8rem;
        margin-top: 0.5rem;
        color: #cbd5e1 !important;
    }
    
    .citation-header {
        font-weight: bold;
        color: #fbbf24;
        font-size: 0.9rem;
    }
    
    /* Button custom styling */
    div.stButton > button {
        background-color: #0284c7 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease;
    }
    
    div.stButton > button:hover {
        background-color: #0369a1 !important;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Cache initialization of Default RAGEngine
# ----------------------------------------------------
@st.cache_resource
def get_default_rag_engine():
    # Initializes default database (loads cached file in seconds if exists)
    return RAGEngine(pdf_path="RBIdoc.pdf", csv_cache_path="text_chunks_and_embeddings_df.csv")

with st.spinner("Initializing Local Regulatory Embeddings Engine..."):
    try:
        default_engine = get_default_rag_engine()
        init_success = True
    except Exception as e:
        init_success = False
        init_error_msg = str(e)

# Create temporary uploads directory in workspace
temp_uploads_dir = os.path.abspath(os.path.join(os.getcwd(), "temp_uploads"))
if not os.path.exists(temp_uploads_dir):
    os.makedirs(temp_uploads_dir)

# ----------------------------------------------------
# Sidebar Controls & API Settings
# ----------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/shield.png", width=80)
    st.title("RBI ReguShield")
    st.markdown("### Compliance Intelligence Platform")
    st.markdown("---")

    # Document Source Selector
    st.markdown("#### 📄 Document Source")
    pdf_source = st.radio(
        "Choose PDF Source:",
        options=["Default RBI Master Directions (2023)", "Upload Custom PDF"],
        index=0
    )
    
    uploaded_file = None
    if pdf_source == "Upload Custom PDF":
        uploaded_file = st.file_uploader("Upload a PDF document:", type=["pdf"])

    st.markdown("---")
    
    # Generation Mode
    st.markdown("#### 🧠 Generation Engine")
    generation_mode = st.radio(
        "Choose LLM Engine:",
        options=["Google Gemini API (Online, High Speed)", "Local CPU Model (Qwen-0.5B - Offline, No Key)"],
        index=0
    )
    use_local = (generation_mode == "Local CPU Model (Qwen-0.5B - Offline, No Key)")
    
    api_key_input = ""
    if not use_local:
        # API Key Configuration
        st.markdown("#### 🔑 Model API Settings")
        env_api_key = os.environ.get("GEMINI_API_KEY", "")
        api_key_input = st.text_input(
            "Enter Google Gemini API Key:",
            value=env_api_key,
            type="password",
            help="Required for the Gemini LLM generative step. You can get one from Google AI Studio."
        )
        
        if api_key_input:
            os.environ["GEMINI_API_KEY"] = api_key_input
        else:
            st.warning("Please enter your Gemini API Key in this field to enable generation.")
    
    # Model parameters
    st.markdown("#### ⚙️ RAG Settings")
    top_k = st.slider("Context Passages (Top-K):", min_value=1, max_value=10, value=5)
    
    # Technical Profile info (Great for resume)
    st.markdown("---")
    st.markdown("#### 🛠️ Technical Architecture Profile")
    st.markdown("""
    - **Embeddings Model**: `sentence-transformers/all-mpnet-base-v2` (768 Dimensions)
    - **Execution Device**: `%s` (Local CPU mode)
    - **Chunk Strategy**: 10 sentences per paragraph chunk (~220 words / ~900 chars)
    - **Local LLM**: `Qwen/Qwen2.5-0.5B-Instruct` (Offline fallback)
    - **Cloud LLM**: `gemini-1.5-flash` via API fallback
    - **Database**: Local Flat Tensor Index (Dot Product similarity)
    """ % ("CUDA GPU" if torch.cuda.is_available() else "CPU"))

# ----------------------------------------------------
# Main Panel Layout
# ----------------------------------------------------
st.title("🛡️ RBI ReguShield")
st.markdown("<p class='subtitle'>Production-grade Local RAG pipeline for financial compliance audits on the RBI Master Directions (NBFC Regulations 2023)</p>", unsafe_allow_html=True)

if not init_success:
    st.error(f"Failed to initialize default embedding engine: {init_error_msg}")
    st.stop()

# ----------------------------------------------------
# Active Engine Resolution
# ----------------------------------------------------
active_engine = default_engine
active_source_title = "RBI NBFC MD"
active_pdf_pages = 328
active_vector_chunks = 468

if pdf_source == "Upload Custom PDF":
    if uploaded_file is not None:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        # If a new file is uploaded or file changes, parse and embed it on-the-fly
        if st.session_state.get("uploaded_file_key") != file_key:
            temp_pdf_path = os.path.join(temp_uploads_dir, "custom.pdf")
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("Processing & embedding custom PDF on CPU (this may take a few seconds)..."):
                try:
                    custom_engine = RAGEngine(pdf_path=temp_pdf_path, csv_cache_path=None)
                    st.session_state.custom_engine = custom_engine
                    st.session_state.uploaded_file_key = file_key
                    st.toast("Custom PDF loaded and indexed successfully!")
                except Exception as e:
                    st.error(f"Error indexing custom PDF: {str(e)}")
                    st.session_state.custom_engine = None
        
        # Resolve metrics and instance
        if st.session_state.get("custom_engine") is not None:
            active_engine = st.session_state.custom_engine
            active_source_title = uploaded_file.name[:12] + "..." if len(uploaded_file.name) > 15 else uploaded_file.name
            try:
                chunks = active_engine.pages_and_chunks
                if chunks:
                    active_pdf_pages = max([c["page_number"] for c in chunks]) + 1
                    active_vector_chunks = len(chunks)
                else:
                    active_pdf_pages = 0
                    active_vector_chunks = 0
            except Exception:
                active_pdf_pages = "N/A"
                active_vector_chunks = "N/A"
    else:
        active_engine = None

# If custom mode is active but no file uploaded, prompt user and stop main layout execution
if active_engine is None:
    st.info("👈 Please upload a PDF document in the sidebar to start asking questions from it, or select 'Default RBI Master Directions (2023)' to query the RBI directions database.")
    st.stop()

# ----------------------------------------------------
# Dashboard Metrics
# ----------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div class='metric-card'><span class='metric-lbl'>Source Doc</span><br><span class='metric-val'>{active_source_title}</span></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><span class='metric-lbl'>PDF Pages</span><br><span class='metric-val'>{active_pdf_pages}</span></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><span class='metric-lbl'>Vector Chunks</span><br><span class='metric-val'>{active_vector_chunks}</span></div>", unsafe_allow_html=True)
with col4:
    st.markdown("<div class='metric-card'><span class='metric-lbl'>Retrieval Speed</span><br><span class='metric-val'>~0.01s</span></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------
# Pre-defined Compliance Queries (Only show for default RBI document)
# ----------------------------------------------------
preset_query = ""
if pdf_source == "Default RBI Master Directions (2023)":
    st.markdown("### 📋 Sample Audit Queries")
    st.write("Click any sample query to instantly search the regulatory database:")
    
    col_q1, col_q2, col_q3 = st.columns(3)
    
    with col_q1:
        if st.button("What is the classification of NBFCs?"):
            preset_query = "What is the classification of NBFCs?"
        if st.button("What is the Net Owned Fund (NOF) requirement?"):
            preset_query = "What is the Net Owned Fund (NOF) requirement?"
    
    with col_q2:
        if st.button("What are Upper Layer regulations?"):
            preset_query = "What are Upper Layer regulations?"
        if st.button("What are the minimum capital adequacy norms?"):
            preset_query = "What are the minimum capital adequacy norms?"
    
    with col_q3:
        if st.button("What is the Leverage Ratio for NBFCs?"):
            preset_query = "What is the Leverage Ratio for NBFCs?"
        if st.button("Explain concentration of credit limits."):
            preset_query = "Explain concentration of credit limits."
    
    st.markdown("---")

# ----------------------------------------------------
# Query Interface
# ----------------------------------------------------
st.markdown("### 💬 Ask Custom PDF Query" if pdf_source == "Upload Custom PDF" else "### 💬 Ask Regulatory Compliance Query")
placeholder_txt = "e.g. What is the main subject of this document?" if pdf_source == "Upload Custom PDF" else "e.g. What are the rules for loans against shares?"
user_query = st.text_input("Enter your query:", value=preset_query, placeholder=placeholder_txt)

if user_query:
    st.markdown("#### Query:")
    st.markdown(f"<div class='chat-bubble chat-user'>❓ <strong>{user_query}</strong></div>", unsafe_allow_html=True)
    
    with st.spinner("Retrieving relevant regulation passages and generating response..."):
        # Run RAG Loop
        answer, context_items = active_engine.query(
            user_query, 
            k=top_k, 
            api_key=api_key_input, 
            use_local_llm=use_local
        )
        
        st.markdown("#### Generated Answer:")
        st.markdown(f"<div class='chat-bubble chat-assistant'>🛡️ <strong>RAG Response:</strong><br><br>{answer}</div>", unsafe_allow_html=True)
        
        # Display Sources / Citations (Critical for Fintech Resume)
        st.markdown("#### 📂 Citations & Context Passages")
        st.write("These passages were retrieved based on vector similarity:")
        
        for idx, item in enumerate(context_items):
            score = item["score"]
            page_num = item["page_number"]
            text_chunk = item["sentence_chunk"]
            
            with st.expander(f"Citation {idx+1}: Page {page_num} | Similarity Score: {score:.4f}"):
                pdf_name = uploaded_file.name if uploaded_file else "RBI NBFC MD"
                st.markdown(f"<div class='citation-header'>{pdf_name} - Page {page_num}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='citation-card'>{text_chunk}</div>", unsafe_allow_html=True)
                pdf_url_path = os.path.abspath(active_engine.pdf_path).replace("\\", "/")
                st.markdown(f"[View Page {page_num} in Original Document](file:///{pdf_url_path}#page={page_num+1})")
