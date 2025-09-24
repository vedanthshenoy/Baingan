import os
from flask import Flask, request
from flask_restx import Api, Resource, fields
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import StorageContext, load_index_from_storage

import faiss

# --------------------------
# Load environment variables
# --------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --------------------------
# Flask + Swagger setup
# --------------------------
app = Flask(__name__)
api = Api(app, version="1.0", title="RAG API", description="RAG with FAISS + HuggingFace embeddings + Groq LLM")
ns = api.namespace("rag", description="RAG operations")

index_model = api.model("IndexRequest", {
    "folder_path": fields.String(required=True, description="Folder containing PDF files")
})

chat_model = api.model("ChatRequest", {
    "system_prompt": fields.String(required=False, description="System prompt"),
    "user_prompt": fields.String(required=True, description="User query")
})

# --------------------------
# Global variables
# --------------------------
query_engine = None
STORAGE_DIR = "storage"  # index + FAISS + docstore will persist here

# --------------------------
# /rag/index endpoint
# --------------------------
@ns.route("/index")
class RagIndex(Resource):
    @ns.expect(index_model)
    def post(self):
        global query_engine

        data = request.get_json()
        folder_path = os.path.normpath(data["folder_path"])

        if not os.path.exists(folder_path):
            return {"error": f"Folder {folder_path} not found"}, 400

        try:
            # Load documents
            documents = SimpleDirectoryReader(folder_path, required_exts=[".pdf"]).load_data()

            # Initialize FAISS vector store
            embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
            dim = len(embed_model.get_text_embedding("test"))

            faiss_index = faiss.IndexFlatL2(dim)
            vector_store = FaissVectorStore(faiss_index=faiss_index)

            # Configure global settings
            Settings.embed_model = embed_model
            Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)

            # Build index
            index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)

            # Persist index
            os.makedirs(STORAGE_DIR, exist_ok=True)
            storage_context = index.storage_context
            storage_context.persist(persist_dir=STORAGE_DIR)

            # Prepare query engine for immediate use
            query_engine = index.as_query_engine()

            return {"message": f"Index built and saved from folder: {folder_path}"}, 200

        except Exception as e:
            return {"error": str(e)}, 500

# --------------------------
# /rag/chat endpoint
# --------------------------
STORAGE_DIR = "./storage"

@ns.route("/chat")
class RagChat(Resource):
    @ns.expect(chat_model)
    def post(self):
        global query_engine

        # Configure embeddings + LLM explicitly at load time
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        llm = Groq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

        Settings.embed_model = embed_model
        Settings.llm = llm

        # Load index from disk if not already loaded
        if query_engine is None:
            try:
                storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
                index = load_index_from_storage(storage_context)
                query_engine = index.as_query_engine()
            except Exception as e:
                return {"error": f"Failed to load index from disk: {str(e)}"}, 500

        # Handle request
        data = request.get_json()
        system_prompt = data.get("system_prompt", "You are a helpful assistant.")
        user_prompt = data.get("user_prompt")

        try:
            full_prompt = f"System: {system_prompt}\nUser: {user_prompt}"
            response = query_engine.query(full_prompt)
            return {"response": str(response)}, 200
        except Exception as e:
            return {"error": str(e)}, 500


# --------------------------
# Run Flask app
# --------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
