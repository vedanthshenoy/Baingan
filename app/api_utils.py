import requests
import json
import google.generativeai as genai
from datetime import datetime
import streamlit as st

def call_api(system_prompt, query, body_template, headers, response_path):
    try:
        safe_system = system_prompt.replace("\n", "\\n").replace("\"", "\\\"")
        safe_query = query.replace("\n", "\\n").replace("\"", "\\\"")

        body = body_template.replace("{system_prompt}", safe_system).replace("{query}", safe_query)
        body_json = json.loads(body)

        response = requests.post(
            st.session_state.api_url,  # Assuming api_url is stored in session or passed
            headers=headers,
            json=body_json,
            timeout=30
        )

        if response.status_code == 200:
            response_data = response.json()
            response_text = response_data
            for key in response_path.split('.'):
                if key in response_text:
                    response_text = response_text[key]
                else:
                    response_text = str(response_data)
                    break
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