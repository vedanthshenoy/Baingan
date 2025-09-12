import streamlit as st
import requests
import json
import google.generativeai as genai
from datetime import datetime
import pandas as pd
import io
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# =========================
# Utility Functions
# =========================

def ensure_prompt_names():
    """Ensure that prompt names align with prompts in session state."""
    prompts = st.session_state.get("prompts", [])
    prompt_names = st.session_state.get("prompt_names", [])

    # If no prompt names, generate them automatically
    if prompts and not prompt_names:
        st.session_state["prompt_names"] = [f"Prompt {i+1}" for i in range(len(prompts))]

    # If too many names, trim to match prompts
    elif len(prompt_names) > len(prompts):
        st.session_state["prompt_names"] = prompt_names[:len(prompts)]

import requests

def call_api(system_prompt, query, *,
             api_url=None, api_key=None,
             response_path=None, timeout=10,
             headers=None, body_template=None):
    """
    Calls the external API and returns a structured response.
    Matches expected unit test behavior.
    """

    # Prepare headers
    final_headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    if headers:
        final_headers.update(headers)

    # Build request payload
    if isinstance(body_template, dict):
        payload = body_template.copy()
    else:
        payload = {}

    payload.setdefault("query", query)
    if system_prompt:
        payload.setdefault("system_prompt", system_prompt)

    try:
        response = requests.post(api_url, headers=final_headers, json=payload, timeout=timeout)

        if response.status_code >= 400:
            return {
                "status": "Error",
                "status_code": response.status_code,
                "response": f"HTTP {response.status_code}: {response.text}"
            }

        data = response.json()

        # Handle dot-separated nested response path, e.g., "data.message"
        if response_path:
            for key in response_path.split("."):
                data = data.get(key, {})

        return {
            "status": "Success",
            "status_code": response.status_code,
            "response": data
        }

    except requests.exceptions.Timeout:
        return {
            "status": "Unknown Error",
            "status_code": "N/A",
            "response": "Connection timeout"
        }
    except requests.exceptions.RequestException as e:
        return {
            "status": "Unknown Error",
            "status_code": "N/A",
            "response": str(e)
        }
    except Exception as e:
        return {
            "status": "Unknown Error",
            "status_code": "N/A",
            "response": str(e)
        }





def suggest_prompt_from_response(target_response, query, gemini_api_key=gemini_api_key):
    """Use Gemini API to generate a better prompt suggestion."""
    if not gemini_api_key:
        return "Gemini API key required for prompt suggestion"

    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        temperature = st.session_state.get("temperature", 50) / 100

        response = model.generate_content(
            f"Given the following query and AI response, suggest a better system prompt.\n\n"
            f"Query: {query}\nResponse: {target_response}",
            generation_config={"temperature": temperature}
        )

        return response.text
    except Exception as e:
        return f"Error generating prompt suggestion: {str(e)}"


# =========================
# UI Functions
# =========================

def render_sidebar():
    """Render the sidebar UI."""
    st.sidebar.header("Controls")
    if st.sidebar.button("Clear All Prompts"):
        for key in [
            "prompts", "prompt_names", "test_results", "chain_results",
            "combination_results", "slider_weights", "last_selected_prompts", "response_ratings"
        ]:
            st.session_state[key] = [] if isinstance(st.session_state.get(key), list) else {}

    st.sidebar.write("Temperature")
    st.session_state["temperature"] = st.sidebar.slider(
        "", 0, 100, st.session_state.get("temperature", 50)
    )


def render_main_area():
    """Render the main app area."""
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Prompts")
        # Prompt management UI would go here
    with col2:
        st.subheader("Results")
        # Results display UI would go here


# =========================
# Main Entrypoint
# =========================

def main():
    st.set_page_config(page_title="Baingan App", layout="wide")
    st.title("Advanced Baingan App")

    # Ensure session state keys exist
    for key, default in [
        ("prompts", []),
        ("prompt_names", []),
        ("test_results", []),
        ("chain_results", []),
        ("combination_results", []),
        ("slider_weights", {}),
        ("last_selected_prompts", []),
        ("response_ratings", {}),
        ("export_data", [])
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # Render UI sections
    render_sidebar()
    render_main_area()


if __name__ == "__main__":
    main()
