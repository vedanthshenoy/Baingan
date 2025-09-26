import requests
import json
import google.generativeai as genai
from datetime import datetime
import streamlit as st
from typing import Dict, Optional, Union, List, Tuple
from urllib.parse import urlencode
# import logging

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

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
    Automatically formats templates and retries with fallback templates on errors.

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

        # Start with the initial template
        current_template = body_template if isinstance(body_template, str) else json.dumps(body_template) if body_template else None
        max_retries = 3
        attempt = 0
        last_error = None

        while attempt < max_retries:
            # Prepare the request body
            body = None
            if current_template:
                if isinstance(current_template, dict):
                    body = current_template
                    if query:
                        body = json.loads(json.dumps(current_template).replace("{query}", query.replace("\n", "\\n").replace("\"", "\\\"")))
                    if system_prompt:
                        body = json.loads(json.dumps(body).replace("{system_prompt}", system_prompt.replace("\n", "\\n").replace("\"", "\\\"")))
                else:
                    safe_query = query.replace("\n", "\\n").replace("\"", "\\\"") if query else ""
                    safe_system = system_prompt.replace("\n", "\\n").replace("\"", "\\\"") if system_prompt else ""
                    body = json.loads(current_template.replace("{query}", safe_query).replace("{system_prompt}", safe_system))

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

            # If success, process and return
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

                # If we had retries, store the successful template for the next run
                if attempt > 0:
                    st.session_state.suggested_body_template = current_template

                return {
                    'response': str(response_text),
                    'status': 'Success' + (' (after retries)' if attempt > 0 else ''),
                    'status_code': response.status_code
                }

            # If error, increment attempt
            else:
                last_error = response.text
                attempt += 1

        # If max retries exceeded
        # logger.error(f"Max retries ({max_retries}) exceeded with error: {last_error}")
        return {
            'response': f"Error: Max retries exceeded - {last_error}",
            'status': 'Error',
            'status_code': response.status_code
        }

    except Exception as e:
        # logger.error(f"Unexpected error: {str(e)}")
        return {
            'response': f"Error: {str(e)}",
            'status': 'Unknown Error',
            'status_code': 'N/A'
        }

def format_request_template(template_str: str) -> Tuple[str, str]:
    """
    Format the request template to convert swagger/postman format to LLM format.
    
    Args:
        template_str: Original template string from user
        
    Returns:
        Tuple of (formatted_template, changes_made_description)
    """
    if not st.session_state.get('gemini_api_key'):
        return template_str, "Gemini API key required for auto-formatting"
    
    try:
        # First validate if it's valid JSON
        try:
            json.loads(template_str)
        except json.JSONDecodeError:
            return template_str, "Invalid JSON format"
        
        # Parse the template as JSON
        try:
            template_json = json.loads(template_str)
        except json.JSONDecodeError:
            return template_str, "Invalid JSON format"

        # Check if the template is already in the correct format
        template_str_test = template_str.replace("{system_prompt}", "test_system").replace("{query}", "test_query")
        try:
            json.loads(template_str_test)
            if "{system_prompt}" in template_str or "{query}" in template_str:
                return template_str, "No changes needed - template already in correct format with proper placeholders"
        except json.JSONDecodeError:
            pass  # Proceed to format the template

        model = genai.GenerativeModel('gemini-2.5-flash')
        
        format_prompt = f"""
You are an API template formatter. Convert this request body template to work with LLM prompting.

Original Template:
{template_str}

Rules for conversion:
1. For fields with "string" values, replace with appropriate placeholders:
   - For system/assistant prompt fields: use "{{system_prompt}}"
   - For user/query fields: use "{{query}}"
   
2. Common field mappings:
   - "system_prompt": "string" → "system_prompt": "{{system_prompt}}"
   - "user_prompt": "string" → "user_prompt": "{{query}}"
   - "query": "string" → "query": "{{query}}"
   - "prompt": "string" → "prompt": "{{query}}"
   - "message": "string" → "message": "{{query}}"
   - "text": "string" → "text": "{{query}}"
   - "input": "string" → "input": "{{query}}"
   - "content": "string" → "content": "{{query}}"

3. If only one prompt field exists:
   - If it's a user/query field (e.g., "user_prompt", "query", "prompt"), keep as "{{query}}"
   - If it's a system field (e.g., "system_prompt"), add a user field with "{{query}}"

4. For other parameters, set reasonable defaults if they are placeholders (e.g., "int", "float", "boolean"):
   - max_tokens: 5000
   - temperature: 0.7
   - top_p: 0.9
   - top_k: 50
   - stream: false

5. Keep existing non-placeholder values (e.g., numbers, booleans) as they are
6. Ensure proper JSON escaping
7. Handle messages array format if present:
   - If "messages" is an array with a single user message, add a system message with "{{system_prompt}}"
   - If "messages" contains a "content": "string", update to "{{query}}"

Output format:
Return ONLY the formatted JSON template, nothing else.

Example transformations:
{{"system_prompt": "string", "user_prompt": "string"}} 
→ {{"system_prompt": "{{system_prompt}}", "user_prompt": "{{query}}"}}

{{"query": "string", "max_tokens": "int"}}
→ {{"query": "{{query}}", "max_tokens": 5000}}

{{"prompt": "string"}}
→ {{"prompt": "{{query}}"}}

{{"messages": [{{"role": "user", "content": "string"}}]}}
→ {{"messages": [{{"role": "system", "content": "{{system_prompt}}"}}, {{"role": "user", "content": "{{query}}"}}]}}
"""
        
        generation_config = genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=1000
        )
        
        try:
            response = model.generate_content(format_prompt, generation_config=generation_config)
            formatted_template = response.text.strip()
        except Exception as e:
            # logger.error(f"Gemini error: {str(e)}")
            # Fallback: Manually transform the template based on rules
            formatted_template = template_str
            changes = []
            
            if isinstance(template_json, dict):
                formatted_json = {}
                prompt_fields = ["system_prompt", "user_prompt", "query", "prompt", "message", "text", "input", "content"]
                has_prompt_field = False
                has_user_field = False
                
                for key, value in template_json.items():
                    if key in prompt_fields and isinstance(value, str) and value.lower() == "string":
                        if key in ["query", "prompt", "message", "text", "input", "content", "user_prompt"]:
                            formatted_json[key] = "{query}"
                            has_user_field = True
                            changes.append(f"Updated '{key}' to use '{{query}}' placeholder")
                        elif key == "system_prompt":
                            formatted_json[key] = "{system_prompt}"
                            has_prompt_field = True
                            changes.append(f"Updated '{key}' to use '{{system_prompt}}' placeholder")
                    elif key == "messages" and isinstance(value, list):
                        formatted_messages = []
                        for msg in value:
                            if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content") == "string":
                                formatted_messages.append({"role": "system", "content": "{system_prompt}"})
                                formatted_messages.append({"role": "user", "content": "{query}"})
                                changes.append("Formatted messages array for LLM conversation")
                            else:
                                formatted_messages.append(msg)
                        formatted_json[key] = formatted_messages
                        has_user_field = True
                    elif key in ["max_tokens", "temperature", "top_p", "top_k", "stream"] and isinstance(value, str):
                        defaults = {"max_tokens": 5000, "temperature": 0.7, "top_p": 0.9, "top_k": 50, "stream": False}
                        formatted_json[key] = defaults[key]
                        changes.append(f"Set default value for '{key}'")
                    else:
                        formatted_json[key] = value
                        if key in prompt_fields:
                            has_prompt_field = True if key == "system_prompt" else has_user_field or True
                
                # If only one prompt field exists, ensure both system and user prompts are included
                if has_prompt_field and not has_user_field and "messages" not in formatted_json:
                    formatted_json["user_prompt"] = "{query}"
                    changes.append("Added 'user_prompt' field with '{query}'")
                elif has_user_field and not has_prompt_field and "messages" not in formatted_json:
                    formatted_json["system_prompt"] = "{system_prompt}"
                    changes.append("Added 'system_prompt' field with '{system_prompt}'")
                
                formatted_template = json.dumps(formatted_json, ensure_ascii=False)
            
            changes_made = "; ".join(changes) if changes else "Template converted to LLM-compatible format"
            return formatted_template, changes_made
        
        # Clean up response - remove code blocks if present
        if formatted_template.startswith('```'):
            lines = formatted_template.split('\n')
            formatted_template = '\n'.join([line for line in lines if not line.strip().startswith('```')])
        
        formatted_template = formatted_template.strip()
        
        # Validate the formatted template
        try:
            test_template = formatted_template.replace("{system_prompt}", "test_system").replace("{query}", "test_query")
            json.loads(test_template)
        except json.JSONDecodeError:
            # logger.error(f"Gemini returned invalid JSON: {formatted_template}")
            # Fallback to manual transformation as above
            formatted_json = {}
            changes = []
            
            if isinstance(template_json, dict):
                prompt_fields = ["system_prompt", "user_prompt", "query", "prompt", "message", "text", "input", "content"]
                has_prompt_field = False
                has_user_field = False
                
                for key, value in template_json.items():
                    if key in prompt_fields and isinstance(value, str) and value.lower() == "string":
                        if key in ["query", "prompt", "message", "text", "input", "content", "user_prompt"]:
                            formatted_json[key] = "{query}"
                            has_user_field = True
                            changes.append(f"Updated '{key}' to use '{{query}}' placeholder")
                        elif key == "system_prompt":
                            formatted_json[key] = "{system_prompt}"
                            has_prompt_field = True
                            changes.append(f"Updated '{key}' to use '{{system_prompt}}' placeholder")
                    elif key == "messages" and isinstance(value, list):
                        formatted_messages = []
                        for msg in value:
                            if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content") == "string":
                                formatted_messages.append({"role": "system", "content": "{system_prompt}"})
                                formatted_messages.append({"role": "user", "content": "{query}"})
                                changes.append("Formatted messages array for LLM conversation")
                            else:
                                formatted_messages.append(msg)
                        formatted_json[key] = formatted_messages
                        has_user_field = True
                    elif key in ["max_tokens", "temperature", "top_p", "top_k", "stream"] and isinstance(value, str):
                        defaults = {"max_tokens": 5000, "temperature": 0.7, "top_p": 0.9, "top_k": 50, "stream": False}
                        formatted_json[key] = defaults[key]
                        changes.append(f"Set default value for '{key}'")
                    else:
                        formatted_json[key] = value
                        if key in prompt_fields:
                            has_prompt_field = True if key == "system_prompt" else has_user_field or True
                
                # If only one prompt field exists, ensure both system and user prompts are included
                if has_prompt_field and not has_user_field and "messages" not in formatted_json:
                    formatted_json["user_prompt"] = "{query}"
                    changes.append("Added 'user_prompt' field with '{query}'")
                elif has_user_field and not has_prompt_field and "messages" not in formatted_json:
                    formatted_json["system_prompt"] = "{system_prompt}"
                    changes.append("Added 'system_prompt' field with '{system_prompt}'")
                
                formatted_template = json.dumps(formatted_json, ensure_ascii=False)
                changes_made = "; ".join(changes) if changes else "Template converted to LLM-compatible format"
                return formatted_template, changes_made
        
        # Determine what changes were made
        changes_made = _describe_template_changes(template_str, formatted_template)
        
        return formatted_template, changes_made
        
    except Exception as e:
        # logger.error(f"Error formatting template: {str(e)}")
        return template_str, f"Error during formatting: {str(e)}"

def _describe_template_changes(original: str, formatted: str) -> str:
    """
    Describe what changes were made to the template.
    """
    try:
        orig_json = json.loads(original.replace("{system_prompt}", "test").replace("{query}", "test"))
        new_json = json.loads(formatted.replace("{system_prompt}", "test").replace("{query}", "test"))
        
        if orig_json == new_json:
            return "No changes needed - template already in correct format"
        
        changes = []
        
        # Check for field additions
        orig_keys = set(orig_json.keys())
        new_keys = set(new_json.keys())
        added_keys = new_keys - orig_keys
        
        if added_keys:
            changes.append(f"Added fields: {', '.join(added_keys)}")
        
        # Check for value changes
        for key in orig_keys & new_keys:
            if str(orig_json[key]) != str(new_json[key]):
                if "system_prompt" in str(new_json[key]) or "query" in str(new_json[key]):
                    changes.append(f"Updated '{key}' to use LLM placeholders")
                else:
                    changes.append(f"Updated '{key}' value")
        
        # Check for messages array changes
        if "messages" in new_json and isinstance(new_json["messages"], list):
            changes.append("Formatted messages array for LLM conversation")
        
        return "; ".join(changes) if changes else "Template converted to LLM-compatible format"
        
    except Exception:
        return "Template converted to LLM-compatible format"

def suggest_prompt_from_response(existing_prompt, target_response, query, rating=None, enhancement_request=None, temperature=50):
    """Generate system prompt suggestions based on existing prompt, target response, rating, and enhancement request using Gemini."""
    if not st.session_state.get('gemini_api_key'):
        return "Gemini API key required for prompt suggestion"
    
    try:
        gemini_temperature = (temperature / 100.0) * 2.0
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Build context based on available information
        context_parts = []
        
        # Always include existing prompt and response
        context_parts.append(f"**Existing System Prompt:**\n{existing_prompt}")
        context_parts.append(f"**Generated Response:**\n{target_response}")
        context_parts.append(f"**Original Query Context:** {query}")
        
        # Add rating context if provided
        if rating is not None:
            rating_percentage = rating * 10  # Convert 0-10 scale to percentage
            if rating_percentage >= 80:
                context_parts.append(f"**User Rating:** {rating}/10 ({rating_percentage}%) - High satisfaction, minor refinements needed")
            elif rating_percentage >= 60:
                context_parts.append(f"**User Rating:** {rating}/10 ({rating_percentage}%) - Moderate satisfaction, improvements needed")
            elif rating_percentage >= 40:
                context_parts.append(f"**User Rating:** {rating}/10 ({rating_percentage}%) - Low satisfaction, significant improvements needed")
            else:
                context_parts.append(f"**User Rating:** {rating}/10 ({rating_percentage}%) - Very low satisfaction, major changes required")
        
        # Add enhancement request if provided
        if enhancement_request and enhancement_request.strip():
            context_parts.append(f"**Requested Enhancements:**\n{enhancement_request}")
        
        context = "\n\n".join(context_parts)
        
        suggestion_prompt = f"""
You are an expert prompt engineer. Based on the provided context, suggest an improved system prompt that addresses the identified issues and enhancement requests.

{context}

**Instructions:**
1. Analyze the existing system prompt and identify areas for improvement
2. Consider the generated response quality and how it aligns with expectations
3. If a rating is provided, factor in the satisfaction level:
   - High ratings (8-10): Make minor refinements and optimizations
   - Medium ratings (6-7): Make moderate improvements to address gaps
   - Low ratings (0-5): Make significant changes to fix major issues
4. If enhancement requests are provided, specifically address those requirements
5. Maintain the core intent and structure while improving clarity, specificity, and effectiveness

**Focus Areas for Improvement:**
- Clarity and specificity of instructions
- Response format and structure guidance  
- Tone and style specifications
- Content depth and coverage requirements
- Any specific behavioral guidelines
- Address the enhancement requests if provided

Return only the improved system prompt without additional explanation or commentary.
"""
        
        generation_config = genai.types.GenerationConfig(
            temperature=gemini_temperature
        )
        
        response = model.generate_content(suggestion_prompt, generation_config=generation_config)
        return response.text
        
    except Exception as e:
        return f"Error generating prompt suggestion: {str(e)}"