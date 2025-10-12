import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
from typing import Optional, List, Dict
import uvicorn
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RAG imports
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import hashlib

# Load environment variables
load_dotenv()

# Load system prompts from YAML file
def load_system_prompts():
    try:
        with open(r"C:\Baingan\test_chat_apps\test_rag\system_prompts.yaml", "r") as file:
            prompts = yaml.safe_load(file)
            logger.info("Loaded system prompts successfully")
            return prompts
    except FileNotFoundError:
        logger.error("system_prompts.yaml file not found")
        raise FileNotFoundError("system_prompts.yaml file not found")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing system_prompts.yaml: {str(e)}")
        raise ValueError(f"Error parsing system_prompts.yaml: {str(e)}")

SYSTEM_PROMPTS = load_system_prompts()

# Initialize FastAPI app
app = FastAPI(
    title="Gemini Chat API with RAG",
    description="FastAPI server with Gemini + optional Retrieval-Augmented Generation (RAG)",
    version="2.0.0"
)

# Request models
class ChatRequest(BaseModel):
    message: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 150
    system_prompt_key: Optional[str] = "default"

class RAGChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    system_prompt_key: Optional[str] = "default"

# Response models
class ChatResponse(BaseModel):
    response: str
    model: str
    status: str

class RAGChatResponse(BaseModel):
    response: str
    sources: Optional[List[Dict]] = None
    model: str
    status: str

# Gemini configuration
MODEL_NAME = "gemini-2.0-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# RAG configuration
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Initialize embedding model + ChromaDB
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        collection = client.get_collection("documents")
    except:
        collection = client.create_collection(name="documents", metadata={"hnsw:space": "cosine"})
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB or embedding model: {str(e)}")
    raise

@app.get("/")
async def root():
    return {
        "message": "Gemini Chat API with RAG is running",
        "model": MODEL_NAME,
        "endpoints": {
            "chat": "/chat",
            "rag_chat": "/chat/rag",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "ok", "gemini_key": bool(GEMINI_API_KEY)}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        system_prompt = SYSTEM_PROMPTS.get(request.system_prompt_key, SYSTEM_PROMPTS.get("default", ""))
        content_text = request.message
        if system_prompt:
            content_text = f"System: {system_prompt}\n\nUser: {request.message}"

        if not GEMINI_API_KEY:
            logger.info(f"Using mock response for /chat with prompt: {system_prompt}")
            return ChatResponse(
                response=f"Mock response for '{request.message}' with system prompt: {system_prompt}",
                model=MODEL_NAME,
                status="mock"
            )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"
        headers = {"Content-Type": "application/json"}

        payload = {
            "contents": [{"parts": [{"text": content_text}]}],
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
                "topP": 0.8,
                "topK": 10
            }
        }

        logger.info(f"Making Gemini API call for /chat with prompt: {system_prompt}")
        response = requests.post(f"{url}?key={GEMINI_API_KEY}", headers=headers, json=payload, timeout=30)

        if response.status_code != 200:
            logger.error(f"Gemini API error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)

        result = response.json()
        generated_text = result["candidates"][0]["content"]["parts"][0]["text"]

        return ChatResponse(response=generated_text, model=MODEL_NAME, status="success")
    except Exception as e:
        logger.error(f"Error in /chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- RAG Functions ----------------
def search_similar_documents(query: str, top_k: int = 5) -> List[Dict]:
    try:
        query_embedding = embedding_model.encode([query]).tolist()[0]
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

        similar_docs = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                similar_docs.append({
                    'content': results['documents'][0][i],
                    'source': results['metadatas'][0][i]['source'],
                    'distance': results['distances'][0][i] if 'distances' in results else 0,
                    'chunk_index': results['metadatas'][0][i]['chunk_index']
                })
        logger.info(f"Retrieved {len(similar_docs)} documents for query: {query}")
        return similar_docs
    except Exception as e:
        logger.error(f"Error in search_similar_documents: {str(e)}")
        return []

def generate_rag_response(query: str, context_docs: List[Dict], system_prompt: str) -> str:
    if not GEMINI_API_KEY:
        logger.info(f"Using mock response for RAG with prompt: {system_prompt}")
        context_summary = " ".join([doc['content'][:50] for doc in context_docs]) if context_docs else "No documents found."
        return f"Mock RAG response for query '{query}' with system prompt '{system_prompt}' and context: {context_summary}"

    context = ""
    for i, doc in enumerate(context_docs, 1):
        context += f"Document {i} ({doc['source']}):\n{doc['content']}\n\n"

    prompt = f"""System: {system_prompt}

Context:
{context}

Question: {query}
Answer:"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"
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

    logger.info(f"Making Gemini API call for RAG with prompt: {system_prompt}")
    response = requests.post(f"{url}?key={GEMINI_API_KEY}", headers=headers, json=payload, timeout=30)
    if response.status_code == 200:
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    else:
        logger.error(f"Gemini API error: {response.status_code} - {response.text}")
        return f"Error: Gemini API returned {response.status_code}: {response.text}"

@app.post("/chat/rag", response_model=RAGChatResponse)
async def rag_chat(request: RAGChatRequest):
    try:
        docs = search_similar_documents(request.query, request.top_k)
        system_prompt = SYSTEM_PROMPTS.get(request.system_prompt_key, SYSTEM_PROMPTS.get("default", ""))
        if not docs:
            logger.info(f"No documents found for query: {request.query}")
            return RAGChatResponse(response="No relevant documents found.", sources=[], model=MODEL_NAME, status="no_docs")
        
        answer = generate_rag_response(request.query, docs, system_prompt)
        return RAGChatResponse(response=answer, sources=docs, model=MODEL_NAME, status="success")
    except Exception as e:
        logger.error(f"Error in /chat/rag endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)