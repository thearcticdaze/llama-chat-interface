# Llama 3.2 3B Chat with Vector Database

This is an enhanced Streamlit application that enables interaction with the Llama 3.2 3B model while using vector embeddings for optimal handling of Excel and CSV data files.

## Features

- **Optimized for Data Analysis**: Specialized processing for Excel and CSV files
- **Vector Embeddings**: Uses sentence-transformers for semantic search capabilities
- **Intelligent Chunking**: Breaks down spreadsheets into meaningful chunks for better retrieval
- **Multi-sheet Excel Support**: Processes all sheets in Excel files separately
- **Statistical Summaries**: Automatically generates statistics for numerical data
- **Semantic Search**: Retrieves the most relevant data for each query

## Requirements

- Python 3.8+
- Ollama installed with the Llama 3.2 3B model
- Libraries listed in requirements.txt

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/macOS
   ```
3. Install the requirements:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Make sure Ollama is running with Llama 3.2 3B model available
   ```
   ollama pull llama3.2:3b
   ```

2. Run the Streamlit app:
   ```
   streamlit run cursorV2.py
   ```

3. In the application:
   - Click "Load Llama 3.2 3B Model" in the sidebar
   - Upload Excel or CSV files using the file uploader
   - Ask questions about the data in the chat interface

## How It Works

1. **Data Processing**:
   - Files are broken down into meaningful chunks (statistics, column info, data rows)
   - Each chunk is converted to a vector embedding

2. **Query Processing**:
   - When you ask a question, it's converted to a vector embedding
   - The system finds the most semantically similar chunks from your uploaded files
   - Only the most relevant chunks are sent to the LLM with your question

3. **Response Generation**:
   - The Llama 3.2 3B model generates a response based on the relevant data chunks
   - Context includes initialization prompt, relevant data, and conversation history

## Tips for Best Results

- **Ask Specific Questions**: The more specific your question, the better the system can find relevant data
- **Mention File Names**: If asking about specific files, mention them by name
- **Prefer Analytical Questions**: Ask for trends, comparisons, or insights rather than just raw data retrieval 