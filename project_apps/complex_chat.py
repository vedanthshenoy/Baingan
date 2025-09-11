import streamlit as st
import chromadb
from chromadb.config import Settings
import requests
import json
import os
from pathlib import Path
import hashlib
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
from typing import List, Dict
import time
from dotenv import load_dotenv
import os

load_dotenv()

# =============================================================================
# CONFIGURATION - MODIFY THESE PATHS AND SETTINGS
# =============================================================================

# Data directory path - CHANGE THIS TO YOUR DATA FOLDER
DATA_DIRECTORY = r"C:\irishman\text_files"  # Change this to your data folder path

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Make sure to set this environment variable
GEMINI_MODEL = "gemini-2.0-flash-exp"

# ChromaDB Configuration
CHROMA_DB_PATH = "./chroma_db"  # Local ChromaDB storage path

# Embedding Model Configuration (using free sentence-transformers)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Free, lightweight embedding model

# =============================================================================
# INITIALIZATION
# =============================================================================

st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize embedding model
@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model for embeddings"""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

# Initialize ChromaDB
@st.cache_resource
def initialize_chromadb():
    """Initialize ChromaDB client and collection"""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Get or create collection
    try:
        collection = client.get_collection("documents")
    except:
        collection = client.create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    return client, collection

# =============================================================================
# DOCUMENT PROCESSING FUNCTIONS
# =============================================================================

def read_text_file(file_path: str) -> str:
    """Read plain text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as file:
            return file.read()

def read_pdf_file(file_path: str) -> str:
    """Read PDF file"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error reading PDF {file_path}: {str(e)}")
    return text

def read_docx_file(file_path: str) -> str:
    """Read DOCX file"""
    try:
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX {file_path}: {str(e)}")
        return ""

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence end
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > start + chunk_size // 2:
                end = start + break_point + 1
                chunk = text[start:end]
        
        chunks.append(chunk.strip())
        start = end - overlap
        
        if start >= len(text):
            break
    
    return [chunk for chunk in chunks if chunk]

def process_documents(data_dir: str) -> List[Dict]:
    """Process all documents in the data directory"""
    documents = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        st.error(f"Data directory not found: {data_dir}")
        return documents
    
    supported_extensions = {'.txt', '.pdf', '.docx', '.md'}
    
    for file_path in data_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                # Read file based on extension
                if file_path.suffix.lower() == '.pdf':
                    content = read_pdf_file(str(file_path))
                elif file_path.suffix.lower() == '.docx':
                    content = read_docx_file(str(file_path))
                else:  # .txt, .md
                    content = read_text_file(str(file_path))
                
                if content.strip():
                    # Chunk the content
                    chunks = chunk_text(content)
                    
                    for i, chunk in enumerate(chunks):
                        doc_id = hashlib.md5(f"{file_path}_{i}".encode()).hexdigest()
                        documents.append({
                            'id': doc_id,
                            'content': chunk,
                            'source': str(file_path.name),
                            'chunk_index': i,
                            'file_path': str(file_path)
                        })
                        
            except Exception as e:
                st.warning(f"Failed to process {file_path.name}: {str(e)}")
    
    return documents

# =============================================================================
# RAG FUNCTIONS
# =============================================================================

def embed_documents(documents: List[Dict], embedding_model, collection):
    """Embed documents and store in ChromaDB"""
    if not documents:
        return
    
    # Check if documents are already embedded
    existing_ids = set()
    try:
        existing_data = collection.get()
        existing_ids = set(existing_data['ids'])
    except:
        pass
    
    new_documents = [doc for doc in documents if doc['id'] not in existing_ids]
    
    if not new_documents:
        return
    
    with st.spinner(f"Embedding {len(new_documents)} new document chunks..."):
        # Create embeddings
        texts = [doc['content'] for doc in new_documents]
        embeddings = embedding_model.encode(texts).tolist()
        
        # Prepare metadata
        metadatas = []
        for doc in new_documents:
            metadatas.append({
                'source': doc['source'],
                'chunk_index': doc['chunk_index'],
                'file_path': doc['file_path']
            })
        
        # Add to ChromaDB
        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=[doc['id'] for doc in new_documents]
        )
    
    st.success(f"Successfully embedded {len(new_documents)} document chunks!")

def search_similar_documents(query: str, embedding_model, collection, top_k: int = 5) -> List[Dict]:
    """Search for similar documents"""
    # Embed the query
    query_embedding = embedding_model.encode([query]).tolist()[0]
    
    # Search in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Format results
    similar_docs = []
    if results['documents'] and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            similar_docs.append({
                'content': results['documents'][0][i],
                'source': results['metadatas'][0][i]['source'],
                'distance': results['distances'][0][i] if 'distances' in results else 0,
                'chunk_index': results['metadatas'][0][i]['chunk_index']
            })
    
    return similar_docs

def generate_response(query: str, context_docs: List[Dict]) -> str:
    """Generate response using Gemini with retrieved context"""
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY not set. Please set the environment variable."
    
    # Prepare context
    context = ""
    for i, doc in enumerate(context_docs, 1):
        context += f"Document {i} (from {doc['source']}):\n{doc['content']}\n\n"
    
    # Prepare prompt
    prompt = f"""You are a helpful AI assistant. Use the following context documents to answer the user's question. If the answer cannot be found in the context, say so clearly.

Context:
{context}

Question: {query}

Answer based on the context provided:"""
    
    # Call Gemini API
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1000,
                "topP": 0.8,
                "topK": 10
            }
        }
        
        response = requests.post(
            f"{url}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return f"Error: Gemini API returned status {response.status_code}: {response.text}"
            
    except Exception as e:
        return f"Error generating response: {str(e)}"

# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    st.title("🤖 RAG Chat Assistant")
    st.markdown("Chat with your documents using Retrieval Augmented Generation")
    
    # Check if API key is set
    if not GEMINI_API_KEY:
        st.error("⚠️ GEMINI_API_KEY environment variable not set!")
        st.code("set GEMINI_API_KEY=your_api_key_here", language="bash")
        return
    
    # Initialize models and database
    try:
        embedding_model = load_embedding_model()
        client, collection = initialize_chromadb()
    except Exception as e:
        st.error(f"Failed to initialize models: {str(e)}")
        return
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Show current data directory
        st.info(f"Data Directory: `{DATA_DIRECTORY}`")
        
        # Process documents button
        if st.button("🔄 Load/Refresh Documents", type="primary"):
            with st.spinner("Processing documents..."):
                documents = process_documents(DATA_DIRECTORY)
                if documents:
                    embed_documents(documents, embedding_model, collection)
                else:
                    st.warning("No documents found or processed.")
        
        # Show collection stats
        try:
            collection_info = collection.get()
            doc_count = len(collection_info['ids']) if collection_info['ids'] else 0
            st.metric("Documents in DB", doc_count)
        except:
            st.metric("Documents in DB", 0)
        
        st.markdown("---")
        
        # Search settings
        st.header("🔍 Search Settings")
        top_k = st.slider("Number of context documents", 1, 10, 5)
        show_sources = st.checkbox("Show source documents", True)
    
    # Main chat interface
    st.header("💬 Chat")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if "sources" in message and show_sources:
                with st.expander("📚 Source Documents"):
                    for i, source in enumerate(message["sources"], 1):
                        st.write(f"**{i}. {source['source']} (chunk {source['chunk_index']})**")
                        st.write(f"Similarity: {1 - source['distance']:.3f}")
                        st.text(source['content'][:200] + "..." if len(source['content']) > 200 else source['content'])
                        st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating response..."):
                # Search for relevant documents
                similar_docs = search_similar_documents(prompt, embedding_model, collection, top_k)
                
                if similar_docs:
                    # Generate response
                    response = generate_response(prompt, similar_docs)
                    st.markdown(response)
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": similar_docs
                    })
                    
                    # Show sources if enabled
                    if show_sources:
                        with st.expander("📚 Source Documents"):
                            for i, source in enumerate(similar_docs, 1):
                                st.write(f"**{i}. {source['source']} (chunk {source['chunk_index']})**")
                                st.write(f"Similarity: {1 - source['distance']:.3f}")
                                st.text(source['content'][:200] + "..." if len(source['content']) > 200 else source['content'])
                                st.markdown("---")
                else:
                    response = "I couldn't find any relevant documents to answer your question. Please make sure documents are loaded."
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

    # Instructions
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. **Set up your data directory**: Modify the `DATA_DIRECTORY` path in the code to point to your documents folder
        2. **Set Gemini API Key**: Set the `GEMINI_API_KEY` environment variable
        3. **Load documents**: Click "Load/Refresh Documents" to process your files
        4. **Start chatting**: Ask questions about your documents
        
        **Supported file formats**: .txt, .pdf, .docx, .md
        
        **Features**:
        - Automatic document chunking
        - Semantic search using embeddings
        - Context-aware responses
        - Source document references
        """)

if __name__ == "__main__":
    main()