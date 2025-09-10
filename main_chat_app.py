from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
from typing import Optional, List, Dict
import uvicorn
from dotenv import load_dotenv
import os

load_dotenv()

# RAG imports
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import hashlib

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
    system_prompt: Optional[str] = None

class RAGChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

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
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
try:
    collection = client.get_collection("documents")
except:
    collection = client.create_collection(name="documents", metadata={"hnsw:space": "cosine"})

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
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")

    try:
        content_text = request.message
        if request.system_prompt:
            content_text = f"System: {request.system_prompt}\n\nUser: {request.message}"

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

        response = requests.post(f"{url}?key={GEMINI_API_KEY}", headers=headers, json=payload, timeout=30)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        result = response.json()
        generated_text = result["candidates"][0]["content"]["parts"][0]["text"]

        return ChatResponse(response=generated_text, model=MODEL_NAME, status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- RAG Functions ----------------
def search_similar_documents(query: str, top_k: int = 5) -> List[Dict]:
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
    return similar_docs

def generate_rag_response(query: str, context_docs: List[Dict]) -> str:
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY not set"

    context = ""
    for i, doc in enumerate(context_docs, 1):
        context += f"Document {i} ({doc['source']}):\n{doc['content']}\n\n"

    prompt = f"""You are a helpful assistant. Use the context to answer.

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

    response = requests.post(f"{url}?key={GEMINI_API_KEY}", headers=headers, json=payload, timeout=30)
    if response.status_code == 200:
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return f"Error: Gemini API returned {response.status_code}: {response.text}"

@app.post("/chat/rag", response_model=RAGChatResponse)
async def rag_chat(request: RAGChatRequest):
    try:
        docs = search_similar_documents(request.query, request.top_k)
        if not docs:
            return RAGChatResponse(response="No relevant documents found.", sources=[], model=MODEL_NAME, status="no_docs")
        
        answer = generate_rag_response(request.query, docs)
        return RAGChatResponse(response=answer, sources=docs, model=MODEL_NAME, status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
