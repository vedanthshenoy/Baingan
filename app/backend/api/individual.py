from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Try to import google.generativeai (Gemini)
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        gemini_available = True
    else:
        gemini_available = False
        gemini_model = None
except Exception:
    genai = None
    gemini_available = False
    gemini_model = None

router = APIRouter()

# Pydantic models
class IndividualTestRequest(BaseModel):
    query: str
    api_config: Dict[str, Any]
    selected_prompts: List[str]  # List of prompt contents

class SuggestPromptRequest(BaseModel):
    response_text: str
    original_query: str

class SavePromptRequest(BaseModel):
    name: str
    content: str
    run_immediately: bool = False
    query: Optional[str] = None
    api_config: Optional[Dict[str, Any]] = None

class UpdateResultRequest(BaseModel):
    unique_id: str
    response: Optional[str] = None
    rating: Optional[int] = None
    remark: Optional[str] = None

# Helper functions
def call_api_helper(system_prompt: str, query: str, body_template: str, headers: Dict[str, str], response_path: str) -> Dict[str, Any]:
    """Helper function to call external API - same logic as original"""
    import requests
    import json
    
    try:
        # Replace placeholders in body template
        body_str = body_template.replace("{system_prompt}", system_prompt)
        body_str = body_str.replace("{query}", query)
        
        # Parse JSON body
        try:
            body = json.loads(body_str)
        except json.JSONDecodeError as e:
            return {
                "response": f"Invalid JSON template: {str(e)}",
                "status": "Failed",
                "status_code": 400
            }
        
        # Make API request
        response = requests.post(headers.get('api_url', ''), json=body, headers=headers, timeout=30)
        
        if response.status_code == 200:
            try:
                response_json = response.json()
                
                # Extract response using path
                response_text = extract_response_by_path(response_json, response_path)
                return {
                    "response": response_text,
                    "status": "Success",
                    "status_code": response.status_code
                }
                
            except json.JSONDecodeError:
                return {
                    "response": response.text,
                    "status": "Success",
                    "status_code": response.status_code
                }
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get('error', error_data.get('message', 'Unknown error'))
                return {
                    "response": f"API Error: {error_msg}",
                    "status": "Failed",
                    "status_code": response.status_code
                }
            except:
                return {
                    "response": f"HTTP {response.status_code}: {response.text}",
                    "status": "Failed",
                    "status_code": response.status_code
                }
                
    except requests.exceptions.RequestException as e:
        return {
            "response": f"Request failed: {str(e)}",
            "status": "Failed",
            "status_code": 500
        }
    except Exception as e:
        return {
            "response": f"Unexpected error: {str(e)}",
            "status": "Failed",
            "status_code": 500
        }

def extract_response_by_path(data: Dict[str, Any], path: str) -> str:
    """Extract response from nested JSON using dot notation path"""
    try:
        keys = path.split('.')
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                raise KeyError(f"Key '{key}' not found in response")
        
        if isinstance(current, (str, int, float)):
            return str(current)
        else:
            import json
            return json.dumps(current)
            
    except Exception as e:
        import json
        return json.dumps(data) if isinstance(data, dict) else str(data)

def save_export_entry(prompt_name: str, system_prompt: str, query: str, response: str, 
                     mode: str, remark: str, status: str, status_code: str, 
                     rating: int, edited: bool) -> str:
    """Save entry to export data and return unique_id - mimics original function"""
    unique_id = str(uuid.uuid4())
    
    # This would normally save to a database or session state
    # For now, we'll just return the unique_id
    return unique_id

def gemini_suggest_func(response_text: str, original_query: str) -> str:
    """Generate prompt suggestion using Gemini"""
    if not gemini_available or not gemini_model:
        return "Gemini is not available. Cannot suggest prompt."

    prompt = f"""
You are given a user query and a response produced by a model.
Query: {original_query}
Response: {response_text}

Your task: Suggest an improved system prompt that would likely produce that kind of response.
Output only the improved system prompt (no extra commentary).
"""
    try:
        result = gemini_model.generate_content(prompt)
        return result.text.strip() if hasattr(result, "text") else str(result)
    except Exception as e:
        return f"Error generating suggestion: {str(e)}"

# Routes
@router.post("/test-all")
async def test_all_prompts(request: IndividualTestRequest):
    """Test all selected prompts individually"""
    results = []
    
    # Build headers
    headers = {"Content-Type": "application/json"}
    api_config = request.api_config
    
    # Extract API URL from config
    api_url = api_config.get('api_url', '')
    if not api_url:
        raise HTTPException(status_code=400, detail="API URL is required")
    
    headers['api_url'] = api_url  # Store for helper function
    
    # Add authentication headers
    auth_type = api_config.get('auth_type', 'None')
    if auth_type == 'Bearer Token' and api_config.get('bearer_token'):
        headers["Authorization"] = f"Bearer {api_config['bearer_token']}"
    elif auth_type == 'API Key' and api_config.get('api_key'):
        key_header = api_config.get('key_header', 'X-API-Key')
        headers[key_header] = api_config['api_key']
    elif auth_type == 'Custom' and api_config.get('custom_headers'):
        headers.update(api_config['custom_headers'])
    
    body_template = api_config.get('body_template', '{"query": "{system_prompt}\\n\\nQuestion: {query}\\nAnswer:", "top_k": 5}')
    response_path = api_config.get('response_path', 'response')
    
    for i, system_prompt in enumerate(request.selected_prompts):
        try:
            # Find prompt name (simplified - in real app would lookup from prompts table)
            prompt_name = f"Prompt {i+1}"
            
            # Call API
            result = call_api_helper(
                system_prompt=system_prompt,
                query=request.query,
                body_template=body_template,
                headers=headers,
                response_path=response_path
            )
            
            response_text = result.get('response', '')
            status = result.get('status', 'Failed')
            status_code = str(result.get('status_code', 'N/A'))
            
            # Save via export entry
            unique_id = save_export_entry(
                prompt_name=prompt_name,
                system_prompt=system_prompt,
                query=request.query,
                response=response_text,
                mode="Individual",
                remark="Saved and ran",
                status=status,
                status_code=status_code,
                rating=0,
                edited=False
            )
            
            result_data = {
                'unique_id': unique_id,
                'test_type': 'Individual',
                'prompt_name': prompt_name,
                'system_prompt': system_prompt,
                'query': request.query,
                'response': response_text,
                'status': status,
                'status_code': status_code,
                'timestamp': datetime.now().isoformat(),
                'rating': 0,
                'remark': 'Saved and ran',
                'edited': False
            }
            
            results.append(result_data)
            
        except Exception as e:
            # Handle individual prompt errors
            unique_id = str(uuid.uuid4())
            error_result = {
                'unique_id': unique_id,
                'test_type': 'Individual',
                'prompt_name': f"Prompt {i+1}",
                'system_prompt': system_prompt,
                'query': request.query,
                'response': f"Error: {str(e)}",
                'status': 'Failed',
                'status_code': '500',
                'timestamp': datetime.now().isoformat(),
                'rating': 0,
                'remark': 'Error occurred',
                'edited': False
            }
            results.append(error_result)
    
    return {"results": results, "total_tested": len(request.selected_prompts)}

@router.post("/suggest-prompt")
async def suggest_prompt(request: SuggestPromptRequest):
    """Generate prompt suggestion using Gemini"""
    suggestion = gemini_suggest_func(request.response_text, request.original_query)
    
    return {
        "suggestion": suggestion,
        "gemini_available": gemini_available
    }

@router.post("/save-suggested-prompt")
async def save_suggested_prompt(request: SavePromptRequest):
    """Save a suggested prompt and optionally run it immediately"""
    
    if request.run_immediately:
        if not request.query or not request.api_config:
            raise HTTPException(status_code=400, detail="Query and API config required for immediate run")
        
        # Build headers
        headers = {"Content-Type": "application/json"}
        api_config = request.api_config
        
        api_url = api_config.get('api_url', '')
        if not api_url:
            raise HTTPException(status_code=400, detail="API URL is required")
        
        headers['api_url'] = api_url
        
        # Add authentication headers
        auth_type = api_config.get('auth_type', 'None')
        if auth_type == 'Bearer Token' and api_config.get('bearer_token'):
            headers["Authorization"] = f"Bearer {api_config['bearer_token']}"
        elif auth_type == 'API Key' and api_config.get('api_key'):
            key_header = api_config.get('key_header', 'X-API-Key')
            headers[key_header] = api_config['api_key']
        elif auth_type == 'Custom' and api_config.get('custom_headers'):
            headers.update(api_config['custom_headers'])
        
        body_template = api_config.get('body_template', '{"query": "{system_prompt}\\n\\nQuestion: {query}\\nAnswer:", "top_k": 5}')
        response_path = api_config.get('response_path', 'response')
        
        # Run the prompt
        try:
            result = call_api_helper(
                system_prompt=request.content,
                query=request.query,
                body_template=body_template,
                headers=headers,
                response_path=response_path
            )
            
            response_text = result.get('response', '')
            status = result.get('status', 'Failed')
            status_code = str(result.get('status_code', 'N/A'))
        except Exception as e:
            response_text = f"Error: {str(e)}"
            status = 'Failed'
            status_code = '500'
        
        # Save to export
        unique_id = save_export_entry(
            prompt_name=request.name,
            system_prompt=request.content,
            query=request.query,
            response=response_text,
            mode="Individual",
            remark="Saved and ran",
            status=status,
            status_code=status_code,
            rating=0,
            edited=False
        )
        
        return {
            "message": f"Saved and ran prompt: {request.name}",
            "unique_id": unique_id,
            "response": response_text,
            "status": status,
            "executed": True
        }
    else:
        # Just save without running
        unique_id = save_export_entry(
            prompt_name=request.name,
            system_prompt=request.content,
            query="",
            response="Prompt saved but not executed",
            mode="Individual",
            remark="Saved only",
            status="Not Executed",
            status_code="N/A",
            rating=0,
            edited=False
        )
        
        return {
            "message": f"Saved prompt: {request.name}",
            "unique_id": unique_id,
            "executed": False
        }

@router.put("/update-result")
async def update_result(request: UpdateResultRequest):
    """Update a test result (response, rating, remark)"""
    
    # In a real app, this would update the database
    # For now, we'll just return success
    updates = {}
    
    if request.response is not None:
        updates['response'] = request.response
        updates['edited'] = True
    
    if request.rating is not None:
        updates['rating'] = request.rating
    
    if request.remark is not None:
        updates['remark'] = request.remark
    
    return {
        "message": "Result updated successfully",
        "unique_id": request.unique_id,
        "updates": updates
    }