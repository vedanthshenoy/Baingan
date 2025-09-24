import requests
import json
import google.generativeai as genai
from datetime import datetime
import streamlit as st
from typing import Dict, Optional, Union, List
from urllib.parse import urlencode

def call_api(
    query: str,
    system_prompt: Optional[str] = None,
    body_template: Optional[Union[str, Dict]] = None,
    headers: Optional[Dict] = None,
    response_path: Optional[str] = None,
    method: str = 'POST',
    query_params: Optional[Dict] = None,
    auth: Optional[Dict] = None,
    api_url: Optional[str] = None
) -> Dict[str, Union[str, int]]:
    """
    Generic function to call any API with flexible parameters, compatible with Streamlit.

    Args:
        query: The main query or user input to send.
        system_prompt: Optional system prompt to include in the request body.
        body_template: Optional string or dict for the request body. Placeholders {query} and {system_prompt} are replaced.
        headers: Optional dictionary of headers (e.g., {"Content-Type": "application/json"}).
        response_path: Optional dot-separated path to extract response data (e.g., "response").
        method: HTTP method (e.g., GET, POST, PUT). Defaults to POST.
        query_params: Optional dictionary of query parameters to append to the URL.
        auth: Optional dictionary for authentication (e.g., {"api_key": "key", "type": "bearer"}).
        api_url: Optional API endpoint URL. If not provided, falls back to st.session_state.api_url.

    Returns:
        Dictionary with 'response', 'status', and 'status_code'.
    """
    try:
        # Use provided api_url or fall back to st.session_state.api_url
        target_url = api_url or st.session_state.get('api_url', '')
        if not target_url:
            return {
                'response': 'Error: API URL not provided or not set in session state',
                'status': 'Error',
                'status_code': 'N/A'
            }

        # Default headers if none provided
        headers = headers or {"Content-Type": "application/json"}
        
        # Handle query parameters
        url = target_url
        if query_params:
            url = f"{target_url}?{urlencode(query_params)}"
        
        # Handle authentication
        if auth:
            if auth.get("type") == "bearer":
                headers["Authorization"] = f"Bearer {auth.get('api_key')}"
            elif auth.get("type") == "query" and query_params is not None:
                query_params["key"] = auth.get("api_key")
                url = f"{target_url}?{urlencode(query_params)}"
            elif auth.get("type") == "header":
                headers["X-API-Key"] = auth.get("api_key")

        # Prepare the request body
        body = None
        if body_template:
            if isinstance(body_template, dict):
                body = body_template
                if query:
                    body = json.loads(json.dumps(body_template).replace("{query}", query.replace("\n", "\\n").replace("\"", "\\\"")))
                if system_prompt:
                    body = json.loads(json.dumps(body).replace("{system_prompt}", system_prompt.replace("\n", "\\n").replace("\"", "\\\"")))
            else:
                safe_query = query.replace("\n", "\\n").replace("\"", "\\\"") if query else ""
                safe_system = system_prompt.replace("\n", "\\n").replace("\"", "\\\"") if system_prompt else ""
                body = json.loads(body_template.replace("{query}", safe_query).replace("{system_prompt}", safe_system))

        # Make the API request
        if method.upper() == 'POST':
            response = requests.post(url, headers=headers, json=body, timeout=30)
        elif method.upper() == 'GET':
            response = requests.get(url, headers=headers, params=body or query_params, timeout=30)
        elif method.upper() == 'PUT':
            response = requests.put(url, headers=headers, json=body, timeout=30)
        elif method.upper() == 'DELETE':
            response = requests.delete(url, headers=headers, json=body, timeout=30)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        # Handle response
        if response.status_code in (200, 201):
            try:
                response_data = response.json()
            except json.JSONDecodeError:
                response_data = {'text': response.text}  # Fallback for non-JSON responses

            response_text = response_data
            if response_path:
                for key in response_path.split('.'):
                    if isinstance(response_text, (dict, list)) and key in response_text:
                        response_text = response_text[key]
                    elif isinstance(response_text, list) and key.isdigit():
                        response_text = response_text[int(key)]
                    else:
                        response_text = str(response_data)
                        break
            else:
                response_text = str(response_data)

            return {
                'response': str(response_text),
                'status': 'Success',
                'status_code': response.status_code
            }
        else:
            return {
                'response': f"Error: {response.text}",
                'status': 'Error',
                'status_code': response.status_code
            }
    except Exception as e:
        return {
            'response': f"Error: {str(e)}",
            'status': 'Unknown Error',
            'status_code': 'N/A'
        }

def suggest_prompt_from_response(target_response, query, temperature=50):
    if not st.session_state.get('gemini_api_key'):
        return "Gemini API key required for prompt suggestion"
    
    try:
        gemini_temperature = (temperature / 100.0) * 2.0
        
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        suggestion_prompt = f"""
Based on the following desired response/output, suggest a system prompt that could generate this type of response when given appropriate queries.

Target Response:
{target_response}

Original Query Context: {query}

Please analyze the response style, tone, structure, and content approach, then suggest a comprehensive system prompt that would guide an AI to produce similar responses. Focus on:
1. Response format and structure
2. Tone and style guidelines
3. Content depth and approach
4. Any specific instructions that seem evident from the output

Return only the suggested system prompt without additional explanation.
"""
        
        generation_config = genai.types.GenerationConfig(
            temperature=gemini_temperature
        )
        
        response = model.generate_content(suggestion_prompt, generation_config=generation_config)
        return response.text
        
    except Exception as e:
        return f"Error generating prompt suggestion: {str(e)}"