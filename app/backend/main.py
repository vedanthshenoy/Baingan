from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import uuid
import pandas as pd
from datetime import datetime
import json

# Load environment variables
load_dotenv()

app = FastAPI(
    title="BainGan API",
    description="Test system prompts, create prompt chains, and combine prompts with AI assistance",
    version="1.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (replace with database in production)
app_state = {
    "prompts": [],
    "prompt_names": [],
    "test_results": [],
    "chain_results": [],
    "combination_results": {},
    "slider_weights": {},
    "response_ratings": {},
    "export_data": []
}

# Pydantic models
class Prompt(BaseModel):
    id: str
    name: str
    content: str

class APIConfig(BaseModel):
    api_url: str
    auth_type: str = "None"
    bearer_token: Optional[str] = None
    api_key: Optional[str] = None
    key_header: str = "X-API-Key"
    custom_headers: Optional[Dict[str, str]] = None
    body_template: str = '{"query": "{system_prompt}\\n\\nQuestion: {query}\\nAnswer:", "top_k": 5}'
    response_path: str = "response"

class TestRequest(BaseModel):
    query: str
    api_config: APIConfig
    prompts: List[str]
    mode: str  # "individual", "chaining", "combination"
    combination_strategy: Optional[str] = None
    temperature: Optional[float] = 0.7
    weights: Optional[Dict[str, float]] = None

class TestResult(BaseModel):
    unique_id: str
    prompt_name: str
    system_prompt: str
    query: str
    response: str
    status: str
    status_code: int
    timestamp: str
    rating: Optional[int] = None
    remark: Optional[str] = None
    edited: bool = False

class UpdateResultRequest(BaseModel):
    unique_id: str
    response: Optional[str] = None
    rating: Optional[int] = None
    remark: Optional[str] = None

# Routes
@app.get("/")
async def root():
    return {"message": "BainGan API is running"}

@app.get("/api/state")
async def get_state():
    """Get current application state"""
    return app_state

@app.post("/api/prompts")
async def add_prompt(prompt: Prompt):
    """Add a new prompt"""
    app_state["prompts"].append({
        "id": prompt.id,
        "name": prompt.name,
        "content": prompt.content
    })
    if prompt.name not in app_state["prompt_names"]:
        app_state["prompt_names"].append(prompt.name)
    return {"message": "Prompt added successfully", "prompt": prompt.dict()}

@app.get("/api/prompts")
async def get_prompts():
    """Get all prompts"""
    return {"prompts": app_state["prompts"], "prompt_names": app_state["prompt_names"]}

@app.put("/api/prompts/{prompt_id}")
async def update_prompt(prompt_id: str, prompt: Prompt):
    """Update an existing prompt"""
    for i, p in enumerate(app_state["prompts"]):
        if p["id"] == prompt_id:
            app_state["prompts"][i] = {
                "id": prompt.id,
                "name": prompt.name,
                "content": prompt.content
            }
            return {"message": "Prompt updated successfully"}
    raise HTTPException(status_code=404, detail="Prompt not found")

@app.delete("/api/prompts/{prompt_id}")
async def delete_prompt(prompt_id: str):
    """Delete a prompt"""
    for i, p in enumerate(app_state["prompts"]):
        if p["id"] == prompt_id:
            deleted_prompt = app_state["prompts"].pop(i)
            # Remove from names if no other prompt has this name
            if not any(p["name"] == deleted_prompt["name"] for p in app_state["prompts"]):
                app_state["prompt_names"].remove(deleted_prompt["name"])
            return {"message": "Prompt deleted successfully"}
    raise HTTPException(status_code=404, detail="Prompt not found")

@app.post("/api/test/individual")
async def test_individual_prompts(request: TestRequest):
    """Test individual prompts"""
    from utils.api_utils import call_api
    
    results = []
    for prompt_content in request.prompts:
        unique_id = str(uuid.uuid4())
        try:
            # Find prompt name
            prompt_name = "Unknown"
            for p in app_state["prompts"]:
                if p["content"] == prompt_content:
                    prompt_name = p["name"]
                    break
            
            # Build headers
            headers = {"Content-Type": "application/json"}
            if request.api_config.auth_type == "Bearer Token" and request.api_config.bearer_token:
                headers["Authorization"] = f"Bearer {request.api_config.bearer_token}"
            elif request.api_config.auth_type == "API Key" and request.api_config.api_key:
                headers[request.api_config.key_header] = request.api_config.api_key
            elif request.api_config.custom_headers:
                headers.update(request.api_config.custom_headers)
            
            # Call API
            response, status_code = call_api(
                request.api_config.api_url,
                request.query,
                request.api_config.body_template,
                headers,
                request.api_config.response_path,
                prompt_content
            )
            
            result = {
                "unique_id": unique_id,
                "prompt_name": prompt_name,
                "system_prompt": prompt_content,
                "query": request.query,
                "response": response,
                "status": "success" if status_code == 200 else "error",
                "status_code": status_code,
                "timestamp": datetime.now().isoformat(),
                "rating": None,
                "remark": None,
                "edited": False
            }
            
            results.append(result)
            app_state["test_results"].append(result)
            
        except Exception as e:
            result = {
                "unique_id": unique_id,
                "prompt_name": prompt_name,
                "system_prompt": prompt_content,
                "query": request.query,
                "response": f"Error: {str(e)}",
                "status": "error",
                "status_code": 500,
                "timestamp": datetime.now().isoformat(),
                "rating": None,
                "remark": None,
                "edited": False
            }
            results.append(result)
            app_state["test_results"].append(result)
    
    return {"results": results}

@app.post("/api/test/chaining")
async def test_prompt_chaining(request: TestRequest):
    """Test prompt chaining"""
    from utils.api_utils import call_api
    
    results = []
    current_query = request.query
    
    for i, prompt_content in enumerate(request.prompts):
        unique_id = str(uuid.uuid4())
        try:
            # Find prompt name
            prompt_name = f"Step {i+1}"
            for p in app_state["prompts"]:
                if p["content"] == prompt_content:
                    prompt_name = p["name"]
                    break
            
            # Build headers
            headers = {"Content-Type": "application/json"}
            if request.api_config.auth_type == "Bearer Token" and request.api_config.bearer_token:
                headers["Authorization"] = f"Bearer {request.api_config.bearer_token}"
            elif request.api_config.auth_type == "API Key" and request.api_config.api_key:
                headers[request.api_config.key_header] = request.api_config.api_key
            elif request.api_config.custom_headers:
                headers.update(request.api_config.custom_headers)
            
            # Call API
            response, status_code = call_api(
                request.api_config.api_url,
                current_query,
                request.api_config.body_template,
                headers,
                request.api_config.response_path,
                prompt_content
            )
            
            result = {
                "unique_id": unique_id,
                "step": i + 1,
                "prompt_name": prompt_name,
                "system_prompt": prompt_content,
                "input_query": current_query,
                "response": response,
                "status": "success" if status_code == 200 else "error",
                "status_code": status_code,
                "timestamp": datetime.now().isoformat(),
                "rating": None,
                "remark": None,
                "edited": False
            }
            
            results.append(result)
            
            # Use this response as input for next step
            if status_code == 200:
                current_query = response
            
        except Exception as e:
            result = {
                "unique_id": unique_id,
                "step": i + 1,
                "prompt_name": prompt_name,
                "system_prompt": prompt_content,
                "input_query": current_query,
                "response": f"Error: {str(e)}",
                "status": "error",
                "status_code": 500,
                "timestamp": datetime.now().isoformat(),
                "rating": None,
                "remark": None,
                "edited": False
            }
            results.append(result)
            break  # Stop chaining on error
    
    app_state["chain_results"].extend(results)
    return {"results": results}

@app.post("/api/test/combination")
async def test_prompt_combination(request: TestRequest):
    """Test prompt combination using AI"""
    try:
        # Import Gemini API
        import google.generativeai as genai
        
        # Configure Gemini
        gemini_key = os.getenv('GEMINI_API_KEY')
        if not gemini_key:
            raise HTTPException(status_code=400, detail="Gemini API key not found")
        
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel('gemini-pro')
        
        # Get prompt contents and names
        prompt_data = []
        for prompt_content in request.prompts:
            prompt_name = "Unknown"
            for p in app_state["prompts"]:
                if p["content"] == prompt_content:
                    prompt_name = p["name"]
                    break
            prompt_data.append({"name": prompt_name, "content": prompt_content})
        
        # Create combination prompt
        combination_prompt = f"""
        Combine the following system prompts to create an optimal response for this query: "{request.query}"
        
        Available prompts:
        """
        
        for i, prompt_info in enumerate(prompt_data, 1):
            weight = request.weights.get(prompt_info["name"], 1.0) if request.weights else 1.0
            combination_prompt += f"\n{i}. {prompt_info['name']} (Weight: {weight}):\n{prompt_info['content']}\n"
        
        combination_prompt += f"""
        
        Strategy: {request.combination_strategy or 'balanced'}
        Temperature: {request.temperature or 0.7}
        
        Create a combined system prompt that leverages the strengths of each prompt according to their weights.
        Then use that combined prompt to answer the query.
        """
        
        # Generate combined response
        response = model.generate_content(combination_prompt)
        
        unique_id = str(uuid.uuid4())
        result = {
            "unique_id": unique_id,
            "prompt_names": [p["name"] for p in prompt_data],
            "combined_prompt": combination_prompt,
            "query": request.query,
            "response": response.text,
            "strategy": request.combination_strategy or 'balanced',
            "temperature": request.temperature or 0.7,
            "weights": request.weights or {},
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "rating": None,
            "remark": None,
            "edited": False
        }
        
        app_state["combination_results"][unique_id] = result
        return {"result": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Combination failed: {str(e)}")

@app.put("/api/results/{result_id}")
async def update_result(result_id: str, request: UpdateResultRequest):
    """Update a test result (rating, remark, response)"""
    # Find and update in test_results
    for result in app_state["test_results"]:
        if result["unique_id"] == result_id:
            if request.response is not None:
                result["response"] = request.response
                result["edited"] = True
            if request.rating is not None:
                result["rating"] = request.rating
            if request.remark is not None:
                result["remark"] = request.remark
            return {"message": "Result updated successfully"}
    
    # Find and update in chain_results
    for result in app_state["chain_results"]:
        if result["unique_id"] == result_id:
            if request.response is not None:
                result["response"] = request.response
                result["edited"] = True
            if request.rating is not None:
                result["rating"] = request.rating
            if request.remark is not None:
                result["remark"] = request.remark
            return {"message": "Result updated successfully"}
    
    # Find and update in combination_results
    if result_id in app_state["combination_results"]:
        result = app_state["combination_results"][result_id]
        if request.response is not None:
            result["response"] = request.response
            result["edited"] = True
        if request.rating is not None:
            result["rating"] = request.rating
        if request.remark is not None:
            result["remark"] = request.remark
        return {"message": "Result updated successfully"}
    
    raise HTTPException(status_code=404, detail="Result not found")

@app.get("/api/export")
async def export_data():
    """Export all test data"""
    export_data = []
    
    # Individual test results
    for result in app_state["test_results"]:
        export_data.append({
            **result,
            "test_type": "individual",
            "step": None,
            "input_query": result["query"],
            "combination_strategy": None,
            "combination_temperature": None,
            "slider_weights": None
        })
    
    # Chain results
    for result in app_state["chain_results"]:
        export_data.append({
            **result,
            "test_type": "chaining",
            "query": result["input_query"],
            "combination_strategy": None,
            "combination_temperature": None,
            "slider_weights": None
        })
    
    # Combination results
    for result in app_state["combination_results"].values():
        export_data.append({
            "unique_id": result["unique_id"],
            "test_type": "combination",
            "prompt_name": ", ".join(result["prompt_names"]),
            "system_prompt": result["combined_prompt"],
            "query": result["query"],
            "response": result["response"],
            "status": result["status"],
            "status_code": 200 if result["status"] == "success" else 500,
            "timestamp": result["timestamp"],
            "edited": result["edited"],
            "step": None,
            "input_query": result["query"],
            "combination_strategy": result["strategy"],
            "combination_temperature": result["temperature"],
            "slider_weights": json.dumps(result["weights"]),
            "rating": result["rating"],
            "remark": result["remark"]
        })
    
    return {"export_data": export_data}

@app.post("/api/suggest-prompt")
async def suggest_prompt(request: dict):
    """Generate prompt suggestions using AI"""
    try:
        from utils.api_utils import suggest_prompt_from_response
        
        response_text = request.get("response", "")
        suggestions = suggest_prompt_from_response(response_text)
        
        return {"suggestions": suggestions}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Suggestion failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)