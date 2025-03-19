import streamlit as st
import ollama
import pandas as pd
import json
import uuid
import tempfile
import os
import time
import chardet
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
import chromadb

# Set page title and layout
st.set_page_config(page_title="Llama 3.2 3B Chat", layout="wide")

st.title("🦙 Llama 3.2 3B Chat Interface")
st.markdown("Upload Excel/CSV files and chat with Llama 3.2 3B model")

# Initialization prompt with enhanced focus on BI and predictive modeling
INIT_PROMPT = """
You are Llama 3.2 3B Chat Assistant.
Your task is to analyze and provide insights into uploaded Excel and CSV files for a mailing company.
Key aspects:
- Identify trends and statistical summaries
- Provide business intelligence insights
- Perform predictive modeling using JSON specifications (e.g., forecasting costs)
- Ensure accuracy and include disclaimers for uncertainty
"""

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'dataframes' not in st.session_state:
    st.session_state.dataframes = {}
if 'file_metadata' not in st.session_state:
    st.session_state.file_metadata = {}

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = load_embedding_model()

if 'vector_db' not in st.session_state:
    client = chromadb.Client()
    st.session_state.vector_db = client.create_collection("data_chunks")

def numpy_to_list(embeddings):
    """Convert numpy array embeddings to list format for ChromaDB."""
    if isinstance(embeddings, np.ndarray):
        return embeddings.tolist()
    elif isinstance(embeddings, list):
        if embeddings and isinstance(embeddings[0], np.ndarray):
            return [e.tolist() for e in embeddings]
    return embeddings

def process_file(uploaded_file):
    """Processes Excel/CSV files and stores them in session state and vector database."""
    file_name = uploaded_file.name
    
    try:
        # Process file into dataframe(s)
        if file_name.endswith(('.xlsx', '.xls')):
            xl = pd.ExcelFile(uploaded_file)
            dfs = {sheet: xl.parse(sheet) for sheet in xl.sheet_names}
            st.session_state.dataframes[file_name] = dfs
            
            # Process each sheet into vector database
            for sheet_name, df in dfs.items():
                process_dataframe_to_vectors(df, file_name, sheet_name)
                
        elif file_name.endswith('.csv'):
            encoding = chardet.detect(uploaded_file.getvalue())['encoding'] or 'utf-8'
            df = pd.read_csv(uploaded_file, encoding=encoding)
            st.session_state.dataframes[file_name] = df
            
            # Process CSV into vector database
            process_dataframe_to_vectors(df, file_name)
            
        st.success(f"{file_name} processed successfully!")
    except Exception as e:
        st.error(f"Error processing {file_name}: {str(e)}")

def process_dataframe_to_vectors(df, file_name, sheet_name=None):
    """Convert dataframe chunks to vector embeddings and store in ChromaDB."""
    try:
        # Create text chunks from DataFrame
        chunks = []
        
        # Add dataframe overview
        overview = f"File: {file_name}"
        if sheet_name:
            overview += f", Sheet: {sheet_name}"
        overview += f"\nColumns: {', '.join(df.columns)}\nRows: {len(df)}"
        chunks.append(overview)
        
        # Add statistical summary if possible
        try:
            stats = df.describe().to_string()
            chunks.append(f"Statistical summary:\n{stats}")
        except:
            pass  # Skip if stats can't be generated
        
        # Add data chunks (process in smaller pieces)
        max_rows = 50  # Process 50 rows at a time
        for i in range(0, len(df), max_rows):
            end_idx = min(i + max_rows, len(df))
            chunk = df.iloc[i:end_idx].to_string()
            chunks.append(f"Data rows {i} to {end_idx-1}:\n{chunk}")
        
        # Generate IDs and metadata for each chunk
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        metadatas = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "file_name": file_name,
                "chunk_type": "overview" if i == 0 else "stats" if i == 1 else "data",
            }
            if sheet_name:
                metadata["sheet_name"] = sheet_name
            metadatas.append(metadata)
        
        # Generate embeddings
        embeddings = st.session_state.embedding_model.encode(chunks)
        
        # Convert numpy arrays to lists for ChromaDB
        embeddings_list = numpy_to_list(embeddings)
        
        # Store in vector database
        st.session_state.vector_db.add(
            documents=chunks,
            embeddings=embeddings_list,
            metadatas=metadatas,
            ids=ids
        )
        
        return len(chunks)
    except Exception as e:
        st.error(f"Error processing dataframe to vectors: {str(e)}")
        return 0

uploaded_files = st.file_uploader("Upload Excel/CSV files", accept_multiple_files=True, type=["xlsx", "xls", "csv"])
if uploaded_files:
    for file in uploaded_files:
        process_file(file)

def get_relevant_context(query, num_results=5):
    """Retrieve relevant chunks from vector database."""
    if 'vector_db' not in st.session_state or st.session_state.vector_db.count() == 0:
        return "No data available. Please upload files first."
    query_embedding = st.session_state.embedding_model.encode(query)
    
    # Convert query embedding to list format
    query_embedding_list = numpy_to_list(query_embedding)
    
    results = st.session_state.vector_db.query(query_embeddings=[query_embedding_list], n_results=num_results)
    return "\n".join(results["documents"][0]) if results["documents"] else "No relevant data found."

def is_ollama_running():
    """Check if Ollama server is running."""
    try:
        response = requests.get("http://localhost:11434/api/version")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def generate_response(prompt):
    """Generates a response using Llama 3.2 3B with statistical insights and predictive modeling."""
    try:
        # Check if Ollama is running
        if not is_ollama_running():
            st.error("Ollama server is not running. Please start Ollama and reload the page.")
            return "Error: Ollama server is not running. Please start Ollama application first."
            
        context = get_relevant_context(prompt)
        st.write(f"Debug - Context retrieved: {len(context)} characters")
        full_prompt = f"{INIT_PROMPT}\n\nRelevant Data:\n{context}\n\nUser: {prompt}"
        st.write(f"Debug - Sending prompt to Llama 3.2 3B")
        
        try:
            response = ollama.generate(model="llama3.2:3b", prompt=full_prompt)
            st.write(f"Debug - Response received from Llama: {len(response['response'])} characters")
            return response["response"]
        except Exception as e:
            if "model 'llama3.2:3b' not found" in str(e).lower():
                st.error("Llama 3.2 3B model not found. Please make sure you have downloaded it in Ollama.")
                return "Error: Llama 3.2 3B model not found. Please run 'ollama pull llama3.2:3b' in command line."
            else:
                raise e
                
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return f"An error occurred: {str(e)}"

# Display chat history
st.subheader("Chat")
for i, (role, message) in enumerate(st.session_state.chat_history):
    if role == "user":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Assistant:** {message}")
    
    # Add a separator between messages
    if i < len(st.session_state.chat_history) - 1:
        st.markdown("---")

user_input = st.text_area("Your message:", height=100)
if st.button("Send") and user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.spinner("Generating response..."):
        response = generate_response(user_input)
    st.session_state.chat_history.append(("assistant", response))
    st.rerun()
