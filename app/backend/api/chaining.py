from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import os
from dotenv import load_dotenv
import requests
import json

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
class ChainingTestRequest(BaseModel):
    query: str
    api_config: Dict[str, Any]
    prompts: List[Dict[str, str]]  # List of {"name": "...", "content": "..."}

class SuggestPromptRequest(BaseModel):
    response_text: str
    original_query: str

class SaveSuggestedPromptRequest(BaseModel):
    name: str
    content: str
    action: str  # "save_only", "save_and_run", "edit"
    query: Optional[str] = None
    api_config: Optional[Dict[str, Any]] = None
    step: Optional[int] = None
    input_query: Optional[str] = None

class UpdateResultRequest(BaseModel):
    unique_id: str
    response: Optional[str] = None
    rating: Optional[int] = None
    remark: Optional[str] = None
    system_prompt: Optional[str] = None  # For reverse prompt feature

# Helper functions
def call_api_helper(system_prompt: str, query: str, body_template: str, headers: Dict[str, str], response_path: str) -> Dict[str, Any]:
    """Helper function to call external API"""
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
        
        # Extract API URL from headers
        api_url = headers.get('api_url', '')
        if not api_url:
            return {
                "response": "API URL not configured",
                "status": "Failed", 
                "status_code": 400
            }
        
        # Remove api_url from headers before request
        request_headers = {k: v for k, v in headers.items() if k != 'api_url'}
        
        # Make API request
        response = requests.post(api_url, json=body, headers=request_headers, timeout=30)
        
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
            return json.dumps(current)
            
    except Exception as e:
        return json.dumps(data) if isinstance(data, dict) else str(data)

def suggest_prompt_from_response(response_text: str, original_query: str) -> str:
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

def save_export_entry(prompt_name: str, system_prompt: str, query: str, response: str, 
                     mode: str, remark: str, status: str, status_code: str, 
                     rating: int, edited: bool, step: int = None, input_query: str = None) -> str:
    """Save entry to export data and return unique_id"""
    unique_id = str(uuid.uuid4())
    # This would normally save to a database
    return unique_id

# Routes
@router.post("/test-chained")
async def test_chained_prompts(request: ChainingTestRequest):
    """Test prompts in sequence (chaining)"""
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
    
    current_query = request.query  # Start with initial query
    
    for step, prompt_data in enumerate(request.prompts, 1):
        prompt_name = prompt_data.get('name', f'Step {step}')
        system_prompt = prompt_data.get('content', '')
        
        # Create step name
        step_name = f"intermediate_result_after_prompt_{prompt_name}" if step < len(request.prompts) else "final_step"
        
        try:
            # Call API
            result = call_api_helper(
                system_prompt=system_prompt,
                query=current_query,
                body_template=body_template,
                headers=headers,
                response_path=response_path
            )
            
            response_text = result.get('response', '')
            status = result.get('status', 'Failed')
            status_code = str(result.get('status_code', 'N/A'))
            
            # Save via export entry
            unique_id = save_export_entry(
                prompt_name=step_name,
                system_prompt=system_prompt,
                query=current_query,
                response=response_text,
                mode="Chaining",
                remark=f"Chained step {step}",
                status=status,
                status_code=status_code,
                rating=0,
                edited=False,
                step=step,
                input_query=request.query
            )
            
            result_data = {
                'unique_id': unique_id,
                'test_type': 'Chaining',
                'prompt_name': step_name,
                'system_prompt': system_prompt,
                'query': current_query,
                'response': response_text,
                'status': status,
                'status_code': status_code,
                'timestamp': datetime.now().isoformat(),
                'rating': 0,
                'remark': f'Chained step {step}',
                'edited': False,
                'step': step,
                'input_query': request.query
            }
            
            results.append(result_data)
            
            # Use response as input for next step
            if status == 'Success' and response_text:
                current_query = response_text
            else:
                # If this step fails, stop the chain
                break
            
        except Exception as e:
            # Handle individual step errors
            unique_id = str(uuid.uuid4())
            error_result = {
                'unique_id': unique_id,
                'test_type': 'Chaining',
                'prompt_name': step_name,
                'system_prompt': system_prompt,
                'query': current_query,
                'response': f"Error: {str(e)}",
                'status': 'Failed',
                'status_code': '500',
                'timestamp': datetime.now().isoformat(),
                'rating': 0,
                'remark': f'Error in step {step}',
                'edited': False,
                'step': step,
                'input_query': request.query
            }
            results.append(error_result)
            break  # Stop chain on error
    
    return {"results": results, "total_steps": len(request.prompts)}

@router.post("/suggest-prompt")
async def suggest_prompt(request: SuggestPromptRequest):
    """Generate prompt suggestion using Gemini"""
    suggestion = suggest_prompt_from_response(request.response_text, request.original_query)
    
    return {
        "suggestion": suggestion,
        "gemini_available": gemini_available
    }

@router.post("/save-suggested-prompt")
async def save_suggested_prompt(request: SaveSuggestedPromptRequest):
    """Save a suggested prompt with different actions"""
    
    if request.action == "save_and_run":
        if not request.query or not request.api_config:
            raise HTTPException(status_code=400, detail="Query and API config required for save_and_run")
        
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
            mode="Chaining",
            remark=f"Saved and ran for step {request.step}",
            status=status,
            status_code=status_code,
            rating=0,
            edited=False,
            step=request.step,
            input_query=request.input_query
        )
        
        return {
            "message": f"Saved and ran prompt: {request.name}",
            "unique_id": unique_id,
            "response": response_text,
            "status": status,
            "executed": True
        }
    else:
        # Just save without running (save_only or edit)
        unique_id = save_export_entry(
            prompt_name=request.name,
            system_prompt=request.content,
            query=request.query or "",
            response="Prompt saved but not executed",
            mode="Chaining",
            remark=f"Save only for step {request.step}",
            status="Not Executed",
            status_code="N/A",
            rating=0,
            edited=False,
            step=request.step,
            input_query=request.input_query
        )
        
        return {
            "message": f"Saved prompt: {request.name}",
            "unique_id": unique_id,
            "executed": False
        }

@router.put("/update-result")
async def update_result(request: UpdateResultRequest):
    """Update a test result (response, rating, remark, system_prompt for reverse prompt)"""
    
    updates = {}
    
    if request.response is not None:
        updates['response'] = request.response
        updates['edited'] = True
    
    if request.rating is not None:
        updates['rating'] = request.rating
    
    if request.remark is not None:
        updates['remark'] = request.remark
    
    if request.system_prompt is not None:
        updates['system_prompt'] = request.system_prompt
        updates['edited'] = True
    
    return {
        "message": "Result updated successfully",
        "unique_id": request.unique_id,
        "updates": updates
    }

@router.post("/reverse-prompt")
async def generate_reverse_prompt(request: UpdateResultRequest):
    """Generate reverse prompt based on edited response"""
    if not request.response:
        raise HTTPException(status_code=400, detail="Response text is required for reverse prompt")
    
    # Use original query from the session or a default
    original_query = "Generate a system prompt for this response"
    suggestion = suggest_prompt_from_response(request.response, original_query)
    
    return {
        "suggested_prompt": suggestion,
        "gemini_available": gemini_available
    }