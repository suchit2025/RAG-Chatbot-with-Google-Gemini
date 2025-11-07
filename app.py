import os
import streamlit as st
import subprocess
import sys

try:
    import google.generativeai as genai
except ImportError:
    st.info("Installing dependencies from requirements.txt...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        st.success("Dependencies installed successfully! Please refresh the page.")
        st.stop()
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to install packages: {e}")
        st.stop()

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile
import torch
import warnings

warnings.filterwarnings('ignore')

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="RAG Chatbot - Gemini API",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CACHED FUNCTIONS ---
@st.cache_resource
def load_embeddings_model():
    with st.spinner("Loading Embeddings Model... (once per session)"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": device}
        )
    return embeddings

@st.cache_resource
def init_gemini(api_key, model_name):
    """Initialize Gemini API"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    return model

# --- CORE LOGIC FUNCTIONS ---
def process_documents(files, c_size, c_overlap):
    if not files:
        st.warning("Please upload at least one PDF file.")
        return None
    with st.spinner("Processing documents..."):
        all_docs = []
        for uploaded_file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            try:
                loader = PDFPlumberLoader(tmp_path)
                docs = loader.load()
                all_docs.extend(docs)
                st.write(f"✓ Parsed {uploaded_file.name} ({len(docs)} pages)")
            except Exception as e:
                st.error(f"Error reading {uploaded_file.name}: {e}")
            finally:
                os.unlink(tmp_path)
        if not all_docs:
            st.warning("No documents were successfully processed.")
            return None
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=c_size,
            chunk_overlap=c_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(all_docs)
        st.info(f"📊 Created {len(chunks)} text chunks from your documents.")
        embeddings = load_embeddings_model()
        vector_store = FAISS.from_documents(chunks, embeddings)
        st.success("✅ Vector store is ready!")
        return vector_store

def qa_with_gemini(vector_store, question, gemini_model):
    """Use Gemini API to answer questions based on retrieved documents"""
    
    # Search for relevant documents
    docs = vector_store.similarity_search(question, k=5)
    
    if not docs:
        return "No relevant information found in documents.", []
    
    # Combine retrieved documents into context
    context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
    
    # Create prompt for Gemini
    prompt = f"""You are a helpful assistant. Answer the following question based ONLY on the provided context.
If the information is not available in the context, say "I don't have this information in the documents."

Context:
{context}

Question: {question}

Answer (be concise and direct):"""
    
    try:
        # Call Gemini API
        response = gemini_model.generate_content(prompt)
        answer = response.text
        return answer, docs
    except Exception as e:
        return f"Error generating response: {str(e)}", docs

# --- SESSION STATE INITIALIZATION ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = None

# --- UI AND MAIN APP LOGIC ---
st.title("📄 RAG Chatbot with Google Gemini API")
st.write("Upload PDFs, process them, and ask questions using Google Gemini AI.")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input("Enter Google Gemini API Key", type="password")
    
    # Model selection
    model_choice = st.selectbox(
        "Choose Gemini Model",
        [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
            "gemini-1.5-flash",
            "gemma-3-1b"
        ]
    )
    
    if api_key:
        st.session_state.gemini_model = init_gemini(api_key, model_choice)
        st.success(f"✅ Connected to {model_choice}!")
    else:
        st.warning("⚠️ Please enter your Gemini API key to proceed.")
    
    if torch.cuda.is_available():
        st.success(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        st.info("ℹ️ Running on CPU (for embeddings only)")
    
    st.subheader("Document Processing")
    chunk_size = st.slider("Chunk Size", 300, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, step=50)
    
    st.info("ℹ️ This uses Google Gemini API for intelligent, accurate responses.")

uploaded_files = st.file_uploader(
    "Upload your PDF documents", type=["pdf"], accept_multiple_files=True
)

if st.button("Process Documents"):
    if not st.session_state.gemini_model:
        st.error("Please enter your Gemini API key first!")
    else:
        st.session_state.vector_store = process_documents(uploaded_files, chunk_size, chunk_overlap)
        if st.session_state.vector_store:
            st.success("Ready to chat! Ask a question below.")
            st.session_state.chat_history = []

st.divider()

if st.session_state.vector_store and st.session_state.gemini_model:
    st.subheader("💬 Chat with your Documents")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if question := st.chat_input("Ask a question about your documents..."):
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("🤖 Gemini is thinking..."):
                try:
                    answer, source_docs = qa_with_gemini(
                        st.session_state.vector_store, 
                        question, 
                        st.session_state.gemini_model
                    )
                    st.markdown(answer)
                    
                    with st.expander("📌 View Sources"):
                        for i, source in enumerate(source_docs, 1):
                            page = source.metadata.get('page', 0) 
                            st.caption(f"Source {i} - Page {page + 1}")
                            st.text(source.page_content[:300] + "...")
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
elif st.session_state.vector_store and not st.session_state.gemini_model:
    st.warning("⚠️ Please enter your Gemini API key in the sidebar to start chatting.")
else:
    st.info("👆 Please enter your Gemini API key and upload PDFs to start chatting.")