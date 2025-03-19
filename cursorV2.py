import streamlit as st
import ollama
import tempfile
import os
import pandas as pd
import chardet
import uuid
from sentence_transformers import SentenceTransformer
import chromadb
from io import StringIO
import numpy as np
import time

# Set page title and layout
st.set_page_config(page_title="Llama 3.2 3B Chat", layout="wide")

# Add a header
st.title("🦙 Llama 3.2 3B Chat Interface")
st.markdown("Upload Excel/CSV files and chat with Llama 3.2 3B model")

# Define the initialization prompt as a constant to ensure consistency
INIT_PROMPT = """
You are Llama 3.2 3B Chat Assistant.
You will be used to chat with a user via a Streamlit interface.
You have access to data from Excel and CSV files uploaded by the user.
Use the provided data snippets as reference to provide context-aware and helpful responses.
Ensure clarity and accuracy in your answers, especially when analyzing numerical data.
Your primary use case will be to analyze and provide data analysis on the uploaded files.
The provided data will be primarily about anything to do with a large scale mailing company (eg. money, transactions, trucks, routes, route numbers, locations, gas, etc).
If you are not certain about the answer, you should ask for more information.
When analyzing data, consider the following aspects:
- Identifying trends and patterns
- Summarizing key statistics
- Providing insights about relationships between variables
- Suggesting potential actions based on the data
"""

# Initialize session state variables if they don't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'file_metadata' not in st.session_state:
    st.session_state.file_metadata = {}

# Initialize embedding model and vector DB
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

if 'embedding_model' not in st.session_state:
    with st.spinner("Loading embedding model..."):
        st.session_state.embedding_model = load_embedding_model()

if 'vector_db' not in st.session_state:
    client = chromadb.Client()
    st.session_state.vector_db = client.create_collection("data_chunks")

# Function to chunk dataframe for processing
def chunk_dataframe(df, chunk_size=100):
    """Split dataframe into chunks with metadata"""
    chunks = []
    metadatas = []
    chunk_texts = []
    
    # Get basic dataframe info for all chunks
    df_info = {
        "columns": list(df.columns),
        "rows": len(df),
        "dtypes": {col: str(df[col].dtype) for col in df.columns}
    }
    
    # Create statistical overview chunk
    try:
        stats_text = f"Statistical summary:\n{df.describe().to_string()}"
        chunks.append(stats_text)
        metadatas.append({
            "type": "statistics",
            "description": "Statistical overview of the entire dataset"
        })
        chunk_texts.append(stats_text)
    except Exception as e:
        pass  # Skip stats if they can't be generated
    
    # Create column info chunk
    columns_text = f"Columns information:\n"
    for col in df.columns:
        # Add column name and data type
        columns_text += f"- {col} ({df[col].dtype})\n"
        # Add unique values for categorical columns with few unique values
        if df[col].dtype == 'object' and df[col].nunique() < 10:
            unique_vals = df[col].unique()
            columns_text += f"  Unique values: {', '.join(str(v) for v in unique_vals[:10])}\n"
        # Add min/max for numeric columns
        elif pd.api.types.is_numeric_dtype(df[col]):
            columns_text += f"  Range: {df[col].min()} to {df[col].max()}\n"
    
    chunks.append(columns_text)
    metadatas.append({
        "type": "schema",
        "description": "Dataset columns and structure information"
    })
    chunk_texts.append(columns_text)
    
    # Chunk the dataframe by rows
    for i in range(0, len(df), chunk_size):
        chunk_df = df.iloc[i:min(i+chunk_size, len(df))]
        chunk_text = chunk_df.to_string()
        
        chunks.append(chunk_text)
        metadatas.append({
            "type": "data_rows",
            "row_range": f"{i} to {min(i+chunk_size, len(df))-1}",
            "row_count": len(chunk_df)
        })
        chunk_texts.append(chunk_text)
    
    return chunks, metadatas, chunk_texts, df_info

# Function to process Excel or CSV file and store as vector embeddings
def process_tabular_file(file_path, file_name):
    """Process Excel/CSV file and store chunks in vector database"""
    # Determine file type and read
    if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        # Read all sheets from Excel
        xl = pd.ExcelFile(file_path)
        sheet_names = xl.sheet_names
        
        all_chunks = []
        all_metadatas = []
        all_embeddings = []
        all_ids = []
        file_info = {
            "file_name": file_name,
            "file_type": "Excel",
            "sheets": {}
        }
        
        # Process each sheet
        for sheet in sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet)
            
            # Skip empty sheets
            if len(df) == 0:
                continue
                
            # Get chunks from this sheet
            chunks, metadatas, chunk_texts, df_info = chunk_dataframe(df)
            
            # Add sheet information to metadatas
            for metadata in metadatas:
                metadata["sheet"] = sheet
                metadata["file_name"] = file_name
            
            # Generate embeddings
            embeddings = st.session_state.embedding_model.encode(chunk_texts)
            ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
            
            # Store sheet info
            file_info["sheets"][sheet] = df_info
            
            # Collect all chunks
            all_chunks.extend(chunks)
            all_metadatas.extend(metadatas)
            all_embeddings.extend(embeddings)
            all_ids.extend(ids)
            
        # Add file overview
        file_overview = f"File {file_name} contains {len(sheet_names)} sheets: {', '.join(sheet_names)}"
        all_chunks.append(file_overview)
        all_metadatas.append({
            "type": "file_overview",
            "file_name": file_name,
            "description": "Excel file overview"
        })
        overview_embedding = st.session_state.embedding_model.encode([file_overview])[0]
        all_embeddings.append(overview_embedding)
        all_ids.append(str(uuid.uuid4()))
            
    else:  # CSV file
        df = pd.read_csv(file_path)
        chunks, metadatas, chunk_texts, df_info = chunk_dataframe(df)
        
        # Add file information to metadatas
        for metadata in metadatas:
            metadata["file_name"] = file_name
        
        # Generate embeddings
        all_embeddings = st.session_state.embedding_model.encode(chunk_texts)
        all_ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        all_chunks = chunks
        all_metadatas = metadatas
        
        file_info = {
            "file_name": file_name,
            "file_type": "CSV",
            "data": df_info
        }
    
    # Store in vector database
    st.session_state.vector_db.add(
        embeddings=all_embeddings,
        documents=all_chunks,
        metadatas=all_metadatas,
        ids=all_ids
    )
    
    return file_info, len(all_chunks)

# Sidebar for model loading and file uploads
with st.sidebar:
    st.header("Settings")
    
    # Model loading section
    st.subheader("Model")
    model_load_button = st.button("Load Llama 3.2 3B Model")
    
    if model_load_button or ('model' in st.session_state):
        if 'model' not in st.session_state:
            with st.spinner("Loading Llama 3.2 3B model... This might take a minute."):
                try:
                    # Set the model name to use with Ollama
                    st.session_state.model = "llama3.2:3b"
                    
                    # Check if model is available in Ollama
                    ollama.show(st.session_state.model)
                    
                    # Initialize with a test prompt to verify the model understands the context
                    response = ollama.generate(
                        model=st.session_state.model, 
                        prompt=INIT_PROMPT + "\nConfirm you understand these instructions.",
                        options={"num_predict": 100}  # Allow response generation
                    )
                    
                    # Store the initialization in session state
                    st.session_state.model_initialized = True
                    st.success("Model loaded and initialized successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
        else:
            st.success("Model is loaded and ready!")
    
    # File upload section
    st.subheader("File Upload")
    uploaded_files = st.file_uploader("Upload Excel or CSV files", 
                                     accept_multiple_files=True, 
                                     type=["xlsx", "xls", "csv"])
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Check if file is already processed
            if uploaded_file.name not in st.session_state.file_metadata:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    start_time = time.time()
                    
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Process and vectorize the file
                        file_info, num_chunks = process_tabular_file(tmp_path, uploaded_file.name)
                        st.session_state.file_metadata[uploaded_file.name] = file_info
                        
                        processing_time = time.time() - start_time
                        st.success(f"File {uploaded_file.name} processed into {num_chunks} searchable chunks! ({processing_time:.2f}s)")
                    except Exception as e:
                        st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
                    finally:
                        # Clean up the temporary file
                        os.unlink(tmp_path)
    
    # Display list of uploaded files
    if st.session_state.file_metadata:
        st.subheader("Uploaded Files")
        for filename, info in st.session_state.file_metadata.items():
            file_type = info.get("file_type", "Unknown")
            
            if file_type == "Excel":
                sheet_names = ", ".join(info.get("sheets", {}).keys())
                st.write(f"📊 {filename} ({len(info.get('sheets', {}))} sheets: {sheet_names})")
            else:  # CSV
                rows = info.get("data", {}).get("rows", "Unknown")
                cols = len(info.get("data", {}).get("columns", []))
                st.write(f"📄 {filename} ({rows} rows, {cols} columns)")
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Clear database button
    if st.button("Clear File Database"):
        if 'vector_db' in st.session_state:
            # Create a new collection to replace the old one
            client = chromadb.Client()
            st.session_state.vector_db = client.create_collection("data_chunks")
        st.session_state.file_metadata = {}
        st.success("File database cleared successfully!")
        st.rerun()

# Main chat interface
st.subheader("Chat")

# Display chat history
for i, (role, message) in enumerate(st.session_state.chat_history):
    if role == "user":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Llama 3.2:** {message}")
    
    # Add a separator between messages
    if i < len(st.session_state.chat_history) - 1:
        st.markdown("---")

# Function to retrieve relevant context from vector database
def get_relevant_context(query, num_results=5):
    """Retrieve most relevant chunks from vector database based on query"""
    if 'vector_db' not in st.session_state or st.session_state.vector_db.count() == 0:
        return "No data available. Please upload files first."
    
    # Generate embedding for query
    query_embedding = st.session_state.embedding_model.encode(query)
    
    # Query the vector database
    results = st.session_state.vector_db.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=num_results
    )
    
    # Format relevant content as context
    if not results['documents'][0]:
        return "No relevant data found in uploaded files."
    
    context = ""
    seen_file_info = set()
    
    # First add file overviews for unique files
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        file_name = metadata.get('file_name', 'Unknown')
        
        # Add file info once per file
        if file_name not in seen_file_info and file_name in st.session_state.file_metadata:
            file_info = st.session_state.file_metadata[file_name]
            if file_info.get("file_type") == "Excel":
                sheet_info = ", ".join(file_info.get("sheets", {}).keys())
                context += f"File {file_name} is an Excel file with sheets: {sheet_info}\n\n"
            else:
                data_info = file_info.get("data", {})
                context += f"File {file_name} is a CSV with {data_info.get('rows', 'unknown')} rows and {len(data_info.get('columns', []))} columns\n\n"
            seen_file_info.add(file_name)
    
    # Then add the actual retrieved chunks
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        file_name = metadata.get('file_name', 'Unknown')
        chunk_type = metadata.get('type', 'data')
        
        # Format based on chunk type
        if chunk_type == "statistics":
            context += f"Statistics from {file_name}:\n{doc}\n\n"
        elif chunk_type == "schema":
            context += f"Column information from {file_name}:\n{doc}\n\n"
        elif chunk_type == "data_rows":
            sheet = metadata.get('sheet', None)
            row_range = metadata.get('row_range', 'unknown range')
            if sheet:
                context += f"Data from {file_name}, sheet '{sheet}', rows {row_range}:\n{doc}\n\n"
            else:
                context += f"Data from {file_name}, rows {row_range}:\n{doc}\n\n"
        else:
            context += f"From {file_name}:\n{doc}\n\n"
    
    return context

# Function to generate response
def generate_response(prompt):
    if 'model' not in st.session_state:
        return "Please load the model first using the button in the sidebar."
    
    # Use the initialization prompt
    init_context = INIT_PROMPT
    
    # Get relevant context from vector database
    file_context = get_relevant_context(prompt, num_results=5)
    
    # Prepare chat history for context
    chat_context = ""
    for role, message in st.session_state.chat_history[-5:]:  # Include last 5 messages for context
        chat_context += f"{'User' if role == 'user' else 'Assistant'}: {message}\n"
    
    # Combine all context with initialization
    full_prompt = f"""
{init_context}

Data retrieved from uploaded files relevant to the question:
{file_context}

Previous conversation:
{chat_context}

User: {prompt}
"""

    # Generate response using Ollama
    response = ollama.generate(
        model=st.session_state.model,
        prompt=full_prompt,
        options={"num_predict": 5000}
    )
    
    return response["response"]

# Input area for user message
user_input = st.text_area("Your message:", height=100)
submit_button = st.button("Send")

# Process user input
if submit_button and user_input:
    # Add user message to chat history
    st.session_state.chat_history.append(("user", user_input))
    
    # Get model response
    with st.spinner("Generating response..."):
        response = generate_response(user_input)
    
    # Add assistant response to chat history
    st.session_state.chat_history.append(("assistant", response))
    
    # Clear input area and refresh to show new messages
    st.rerun()