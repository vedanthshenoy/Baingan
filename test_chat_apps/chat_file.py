from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
from typing import Optional
import uvicorn
from dotenv import load_dotenv
import os

load_dotenv()
# Initialize FastAPI app
app = FastAPI(
    title="Gemini Chat API",
    description="Simple FastAPI server to interface with Google Gemini 2.0 Flash model",
    version="1.0.0"
)

# Request model
class ChatRequest(BaseModel):
    message: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 150
    system_prompt: Optional[str] = None

# Response model
class ChatResponse(BaseModel):
    response: str
    model: str
    status: str

# Gemini configuration
MODEL_NAME = "gemini-2.0-flash-exp"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Set this as environment variable

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Gemini Chat API is running",
        "model": MODEL_NAME,
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Check if Gemini API key is configured"""
    if not GEMINI_API_KEY:
        return {
            "status": "unhealthy",
            "gemini_available": False,
            "error": "GEMINI_API_KEY environment variable not set"
        }
    
    try:
        # Test API key with a simple request
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"
        headers = {
            "Content-Type": "application/json",
        }
        
        test_payload = {
            "contents": [{"parts": [{"text": "Hello"}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 10
            }
        }
        
        response = requests.post(
            f"{url}?key={GEMINI_API_KEY}",
            headers=headers,
            json=test_payload,
            timeout=10
        )
        
        if response.status_code == 200:
            return {
                "status": "healthy",
                "gemini_available": True,
                "model": MODEL_NAME
            }
        else:
            return {
                "status": "unhealthy",
                "gemini_available": False,
                "error": f"Gemini API returned status {response.status_code}: {response.text}"
            }
            
    except requests.exceptions.RequestException as e:
        return {
            "status": "unhealthy",
            "gemini_available": False,
            "error": f"Connection error: {str(e)}"
        }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint that sends message to Gemini 2.0 Flash model
    """
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY environment variable not set. Please set your API key."
        )
    
    try:
        # Prepare the content
        content_text = request.message
        if request.system_prompt:
            content_text = f"System instructions: {request.system_prompt}\n\nUser: {request.message}"
        
        # Prepare Gemini API request
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"
        headers = {
            "Content-Type": "application/json",
        }
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": content_text}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
                "topP": 0.8,
                "topK": 10
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
        
        # Make request to Gemini API
        response = requests.post(
            f"{url}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            error_detail = f"Gemini API error (Status {response.status_code}): {response.text}"
            raise HTTPException(
                status_code=response.status_code,
                detail=error_detail
            )
        
        # Parse response
        gemini_response = response.json()
        
        # Extract the generated text with better error handling
        try:
            candidates = gemini_response.get("candidates", [])
            if not candidates:
                raise HTTPException(
                    status_code=500,
                    detail="No candidates in Gemini response"
                )
            
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            
            if not parts:
                raise HTTPException(
                    status_code=500,
                    detail="No parts in Gemini response content"
                )
            
            generated_text = parts[0].get("text", "").strip()
            
        except (KeyError, IndexError, AttributeError) as e:
            # Log the full response for debugging
            print(f"Gemini response structure error: {e}")
            print(f"Full response: {json.dumps(gemini_response, indent=2)}")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected response structure from Gemini API. Response: {str(gemini_response)[:500]}"
            )
        
        if not generated_text:
            raise HTTPException(
                status_code=500,
                detail="Empty response from Gemini model"
            )
        
        return ChatResponse(
            response=generated_text,
            model=MODEL_NAME,
            status="success"
        )
        
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Request to Gemini API timed out"
        )
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Gemini API. Check your internet connection."
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Request error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/chat/simple")
async def simple_chat(request: dict):
    """
    Simplified chat endpoint that accepts any JSON with 'message' field
    Compatible with the Streamlit app format
    """
    try:
        message = request.get("message")
        if not message:
            raise HTTPException(
                status_code=400,
                detail="Missing 'message' field in request"
            )
        
        # Convert to ChatRequest format
        chat_request = ChatRequest(
            message=message,
            temperature=request.get("temperature", 0.7),
            max_tokens=request.get("max_tokens", 150),
            system_prompt=request.get("system_prompt")
        )
        
        # Use the main chat endpoint
        result = await chat(chat_request)
        
        # Return in a simple format
        return {
            "response": result.response,
            "model": result.model
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

if __name__ == "__main__":
    print("🚀 Starting Gemini Chat FastAPI Server...")
    print(f"📡 API URL: http://localhost:8000")
    print(f"📚 API Docs: http://localhost:8000/docs")
    print(f"🔗 Chat Endpoint: http://localhost:8000/chat")
    print(f"🔗 Simple Chat Endpoint: http://localhost:8000/chat/simple")
    print(f"🤖 Model: {MODEL_NAME}")
    print(f"🔑 API Key: {'✅ Set' if GEMINI_API_KEY else '❌ Not Set'}")
    
    if not GEMINI_API_KEY:
        print("\n⚠️  WARNING: GEMINI_API_KEY environment variable not set!")
        print("Set it with: set GEMINI_API_KEY=your_api_key_here (Windows)")
        print("Or: export GEMINI_API_KEY=your_api_key_here (Linux/Mac)")
    
    print("\n" + "="*50)
    print("FOR STREAMLIT APP USE:")
    print("API Endpoint URL: http://localhost:8000/chat/simple")
    print("Request Body Template:")
    print('{\n  "message": "{prompt}",\n  "temperature": 0.7,\n  "max_tokens": 150\n}')
    print("Response Text Path: response")
    print("="*50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )