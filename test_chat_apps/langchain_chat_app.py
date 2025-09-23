import os
import dotenv
from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load environment variables from a .env file
dotenv.load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Load the Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found. Please create a .env file with your API key.")
    print("Example: GOOGLE_API_KEY='your_api_key_here'")

# --- Globals for the RAG pipeline ---
db = None
qa_chain = None
indexed_folder_path = None
indexed_file_type = None

# A mapping of file extensions to their corresponding LangChain loader classes
LOADER_MAPPING = {
    "txt": TextLoader,
    "pdf": PyPDFLoader,
}

def initialize_vector_store(folder_path, file_extension):
    """Initializes the Chroma vector store from a directory of documents."""
    print(f"Loading documents from directory {folder_path}...")
    
    loader_cls = LOADER_MAPPING.get(file_extension.lower())
    if not loader_cls:
        print(f"Error: Unsupported file extension '{file_extension}'. Supported types are: {', '.join(LOADER_MAPPING.keys())}")
        return False

    try:
        loader = DirectoryLoader(
            path=folder_path,
            glob=f"**/*.{file_extension}",
            show_progress=True,
            loader_cls=loader_cls
        )
        documents = loader.load()
    except Exception as e:
        print(f"Error loading documents from directory: {e}")
        return False
    
    if not documents:
        print(f"No '.{file_extension}' files found in the directory.")
        return False

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} document(s) into {len(docs)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Updated to use HuggingFaceEmbeddings
    global db
    db = Chroma.from_documents(docs, embeddings)
    print("ChromaDB vector store created.")
    return True

def initialize_qa_chain():
    """Initializes the retrieval chain with the Gemini model and prompt template."""
    global qa_chain
    if not GOOGLE_API_KEY:
        print("API key is missing. Cannot initialize the QA chain.")
        return

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=GOOGLE_API_KEY)

    # Define the prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are an AI assistant for question-answering tasks. Use the following context to answer the question. 
    If you don't know the answer, just say that you don't know. Keep the answer concise.

    Context:
    {context}

    Question:
    {input}
    """)
    
    # Create the document combining chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create the retrieval chain
    retriever = db.as_retriever()
    qa_chain = create_retrieval_chain(retriever, document_chain)
    print("Retrieval chain initialized.")

@app.route('/api/index', methods=['POST'])
def index_document():
    """
    POST endpoint to index a directory of documents.

    Request JSON format:
    {
        "folder_path": "path/to/your/documents/folder",
        "file_extension": "pdf" 
    }
    """
    data = request.get_json()
    if 'folder_path' not in data or 'file_extension' not in data:
        return jsonify({"error": "Missing 'folder_path' or 'file_extension' in request body."}), 400

    folder_path = data['folder_path']
    file_extension = data['file_extension']
    
    if file_extension.lower() not in LOADER_MAPPING:
        return jsonify({"error": f"Unsupported file extension '{file_extension}'. Supported types are: {', '.join(LOADER_MAPPING.keys())}"}), 400

    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return jsonify({"error": f"Folder not found or is not a directory at {folder_path}"}), 404

    global indexed_folder_path
    global indexed_file_type
    
    # Check if the same folder and file type are already indexed
    if indexed_folder_path == folder_path and indexed_file_type == file_extension:
        return jsonify({"message": f"Folder '{os.path.basename(folder_path)}' with file type '.{file_extension}' is already indexed."}), 200

    if initialize_vector_store(folder_path, file_extension):
        initialize_qa_chain()
        indexed_folder_path = folder_path
        indexed_file_type = file_extension
        return jsonify({"message": f"Successfully indexed documents from folder '{os.path.basename(folder_path)}' of type '.{file_extension}'."}), 200
    else:
        return jsonify({"error": f"Failed to index folder. Check the folder path and file contents."}), 500

@app.route('/api/rag', methods=['POST'])
def process_rag_query():
    """
    POST endpoint to answer a query using the indexed documents.

    Request JSON format:
    {
        "query": "What is the main topic of the documents?"
    }
    """
    data = request.get_json()
    if 'query' not in data:
        return jsonify({"error": "Missing 'query' in request body."}), 400

    if qa_chain is None:
        return jsonify({"error": "No documents indexed. Please use the /api/index endpoint first."}), 400

    query = data['query']
    print(f"Processing query: '{query}'")

    try:
        result = qa_chain.invoke({"input": query})
        response_text = result.get('answer', 'No answer found.')
        return jsonify({"response": response_text}), 200
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('documents'):
        os.makedirs('documents')
        print("Created 'documents' directory. Please place your documents inside it.")

    print("Flask app is running. Use a tool like curl or Postman to test the API.")
    print("First, index a document using the /api/index endpoint, specifying the 'folder_path' and 'file_extension'.")
    print("Example Request Body for /api/index:")
    print('{ "folder_path": "C:\\irishman\\text_files", "file_extension": "pdf" }')
    print("Then, send a query to the /api/rag endpoint.")
    app.run(debug=False, threaded=True)  # Changed to debug=False and added threaded=True