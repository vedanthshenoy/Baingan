import streamlit as st
import pandas as pd
import google.generativeai as genai
from io import BytesIO
import time
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Configure page
st.set_page_config(
    page_title="Prompt Testing & Improvement",
    page_icon="🔍",
    layout="wide"
)

# Load environment variables from specific path
load_dotenv("C:\\Baingan\\.env")

def validate_gemini_key(api_key: str) -> bool:
    """Validate if the Gemini API key is working."""
    if not api_key:
        return False
    try:
        genai.configure(api_key=api_key)
        # Try to create a model (this will fail if the key is invalid)
        model = genai.GenerativeModel('gemini-2.5-flash')
        return True
    except Exception:
        return False

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'feedback' not in st.session_state:
    st.session_state.feedback = {}
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = {}
if 'gemini_key' not in st.session_state:
    st.session_state.gemini_key = os.getenv('GEMINI_API_KEY')
if 'test_function' not in st.session_state:
    st.session_state.test_function = None

# Handle Gemini API key setup in sidebar
with st.sidebar:
    if not st.session_state.gemini_key or not validate_gemini_key(st.session_state.gemini_key):
        st.warning("Please enter your Gemini API key to continue.")
        api_key = st.text_input("Enter your Gemini API key:", type="password", key="gemini_api_input")
        if api_key:
            if validate_gemini_key(api_key):
                st.session_state.gemini_key = api_key
                st.success("API key validated successfully!")
                st.rerun()
            else:
                st.error("Invalid API key. Please check and try again.")
                st.stop()
    else:
        st.success("Gemini API key is configured")



def configure_gemini(api_key: str):
    """Configure Gemini API with the provided key."""
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error configuring Gemini: {str(e)}")
        return False


def get_gemini_suggestion(original_prompt: str, current_output: str, 
                         user_feedback: str, rating: int) -> str:
    """
    Use Gemini to suggest an improved system prompt based on user feedback.
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        prompt = f"""You are an expert at improving system prompts for AI applications.

Original System Prompt: {original_prompt}
Current Output: {current_output}
User Rating: {rating}/10
User Feedback: {user_feedback}

The system prompt may use placeholders like {{context}}, {{user_prompt}}, etc. that get filled with actual values (RAG context, user queries, etc.).

Based on the user's feedback and rating, suggest an IMPROVED system prompt that:
1. Addresses the user's concerns from their feedback
2. Maintains the same placeholders ({{context}}, {{user_prompt}}, etc.)
3. Is more effective, clear, or appropriate based on the feedback
4. Improves instruction clarity and output quality

IMPORTANT: Return ONLY the improved system prompt, nothing else. No explanations, no additional text.

Improved System Prompt:"""

        response = model.generate_content(prompt)
        suggested_prompt = response.text.strip()
        
        # Clean up any markdown or extra formatting
        suggested_prompt = suggested_prompt.replace('```', '').strip()
        
        return suggested_prompt
    
    except Exception as e:
        st.error(f"Error getting Gemini suggestion: {str(e)}")
        return None


def execute_function_with_prompt(system_prompt: str, user_prompt: str = None) -> str:
    """
    Execute the user's test function with the new system prompt.
    This is a placeholder - user needs to provide their actual function.
    """
    if st.session_state.test_function:
        try:
            # User should set this in their code
            result = st.session_state.test_function(system_prompt, user_prompt)
            return result
        except Exception as e:
            return f"Error executing function: {str(e)}"
    else:
        # Mock implementation for demonstration
        context = "Sample retrieved context from RAG"
        try:
            formatted_prompt = system_prompt.format(context=context, user_prompt=user_prompt or "")
            return f"[Mock Output] {formatted_prompt}"
        except Exception as e:
            return f"Error formatting prompt: {str(e)}"


def convert_df_to_excel(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to Excel file bytes."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
    return output.getvalue()


def main():
    st.title("🔍 Prompt Testing & Improvement App")
    st.markdown("Upload your test results, provide feedback, and get AI-powered prompt improvements!")
    
    # Sidebar for API key and settings
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.gemini_key or "",
            help="Enter your Google Gemini API key"
        )
        
        if api_key and api_key != st.session_state.gemini_key:
            if configure_gemini(api_key):
                st.session_state.gemini_key = api_key
                st.success("✅ Gemini configured successfully!")
            else:
                st.error("❌ Failed to configure Gemini")
        
        st.markdown("---")
        st.markdown("### 📖 Instructions")
        st.markdown("""
        1. Upload your test results CSV/Excel
        2. Review each system prompt and output
        3. Provide feedback and rating
        4. Get AI-powered prompt improvements
        5. Test new prompts
        6. Download updated results
        """)
        
        st.markdown("---")
        st.markdown("### 💡 About")
        st.markdown("""
        This app helps you iterate on:
        - **System Prompts**: Instructions for AI models
        - **User Prompts**: Query inputs
        - **Context**: RAG retrievals or other data
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Test Results")
        uploaded_file = st.file_uploader(
            "Upload your DataFrame (CSV or Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="Upload the DataFrame from your test decorator"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.df = df
                st.success(f"✅ Loaded {len(df)} test results")
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    with col2:
        if st.session_state.df is not None:
            st.subheader("📊 Data Overview")
            st.metric("Total Tests", len(st.session_state.df))
            st.metric("Columns", len(st.session_state.df.columns))
            
            # Show column names
            cols = st.session_state.df.columns.tolist()
            st.write("**Columns:**", ", ".join(cols))
    
    # Main feedback section
    if st.session_state.df is not None:
        st.markdown("---")
        st.header("📝 Review & Provide Feedback")
        
        df = st.session_state.df
        
        # Identify columns (flexible to handle different naming)
        prompt_col = None
        output_col = None
        
        for col in df.columns:
            if 'system' in col.lower() or 'prompt' in col.lower():
                prompt_col = col
            elif 'output' in col.lower() or 'result' in col.lower():
                output_col = col
        
        if not prompt_col or not output_col:
            st.error("⚠️ Could not identify prompt and output columns. Please ensure your DataFrame has appropriate column names.")
            st.info("Expected columns like: 'system_prompt', 'func_output' or similar")
            return
        
        # Add columns for tracking if they don't exist
        if 'rating' not in df.columns:
            df['rating'] = 0
        if 'feedback' not in df.columns:
            df['feedback'] = ""
        if 'suggested_prompt' not in df.columns:
            df['suggested_prompt'] = ""
        if 'new_output' not in df.columns:
            df['new_output'] = ""
        
        # Display each row with feedback options
        for idx, row in df.iterrows():
            with st.expander(f"**Test #{idx + 1}** - Click to review and provide feedback", expanded=(idx == 0)):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("##### 📋 Original System Prompt")
                    st.code(row[prompt_col], language="text")
                    
                    st.markdown("##### 📤 Current Output")
                    st.info(row[output_col])
                
                with col2:
                    st.markdown("##### 💬 Your Feedback")
                    
                    # Rating slider
                    rating = st.slider(
                        "Rate this output (0-10)",
                        min_value=0,
                        max_value=10,
                        value=int(row['rating']) if pd.notna(row['rating']) else 5,
                        key=f"rating_{idx}",
                        help="0 = Poor, 10 = Excellent"
                    )
                    
                    # Feedback text
                    feedback_text = st.text_area(
                        "What improvements would you like?",
                        value=row['feedback'] if pd.notna(row['feedback']) else "",
                        placeholder="E.g., Be more concise, add structure, improve clarity...",
                        key=f"feedback_{idx}",
                        height=100
                    )
                    
                    # Update DataFrame
                    df.at[idx, 'rating'] = rating
                    df.at[idx, 'feedback'] = feedback_text
                    
                    # Get AI suggestion button
                    if st.button(f"🤖 Get AI Suggestion", key=f"suggest_{idx}"):
                        if not st.session_state.gemini_key:
                            st.warning("⚠️ Please enter your Gemini API key in the sidebar first")
                        elif not feedback_text:
                            st.warning("⚠️ Please provide feedback about what you'd like to improve")
                        else:
                            with st.spinner("Getting AI suggestion..."):
                                suggestion = get_gemini_suggestion(
                                    row[prompt_col],
                                    row[output_col],
                                    feedback_text,
                                    rating
                                )
                                
                                if suggestion:
                                    df.at[idx, 'suggested_prompt'] = suggestion
                                    st.success("✅ Got AI suggestion!")
                                    st.rerun()
                
                # Show suggested prompt if available
                if pd.notna(row['suggested_prompt']) and row['suggested_prompt']:
                    st.markdown("---")
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("##### 🎯 AI Suggested System Prompt")
                        st.code(row['suggested_prompt'], language="text")
                    
                    with col2:
                        if st.button(f"▶️ Test This Prompt", key=f"test_{idx}"):
                            with st.spinner("Running test..."):
                                new_output = execute_function_with_prompt(row['suggested_prompt'])
                                df.at[idx, 'new_output'] = new_output
                                st.success("✅ Test completed!")
                                st.rerun()
                    
                    # Show new output if available
                    if pd.notna(row['new_output']) and row['new_output']:
                        st.markdown("##### 🎉 New Output")
                        st.success(row['new_output'])
                        
                        # Option to use this as the new prompt
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"✅ Use This Prompt", key=f"use_{idx}"):
                                df.at[idx, prompt_col] = row['suggested_prompt']
                                df.at[idx, output_col] = row['new_output']
                                df.at[idx, 'suggested_prompt'] = ""
                                df.at[idx, 'new_output'] = ""
                                st.success("✅ Prompt updated!")
                                st.rerun()
                        
                        with col2:
                            if st.button(f"❌ Discard", key=f"discard_{idx}"):
                                df.at[idx, 'suggested_prompt'] = ""
                                df.at[idx, 'new_output'] = ""
                                st.info("Suggestion discarded")
                                st.rerun()
        
        # Update session state
        st.session_state.df = df
        
        # Batch operations
        st.markdown("---")
        st.header("🚀 Batch Operations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🤖 Get All Suggestions", use_container_width=True):
                if not st.session_state.gemini_key:
                    st.warning("⚠️ Please enter your Gemini API key first")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, row in df.iterrows():
                        if pd.notna(row['feedback']) and row['feedback'] and not row['suggested_prompt']:
                            status_text.text(f"Processing test {idx + 1}/{len(df)}...")
                            
                            suggestion = get_gemini_suggestion(
                                row[prompt_col],
                                row[output_col],
                                row['feedback'],
                                int(row['rating'])
                            )
                            
                            if suggestion:
                                df.at[idx, 'suggested_prompt'] = suggestion
                            
                            progress_bar.progress((idx + 1) / len(df))
                            time.sleep(0.5)  # Rate limiting
                    
                    st.session_state.df = df
                    status_text.text("✅ All suggestions generated!")
                    st.rerun()
        
        with col2:
            if st.button("▶️ Test All Suggestions", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                count = 0
                for idx, row in df.iterrows():
                    if pd.notna(row['suggested_prompt']) and row['suggested_prompt']:
                        status_text.text(f"Testing {idx + 1}/{len(df)}...")
                        
                        new_output = execute_function_with_prompt(row['suggested_prompt'])
                        df.at[idx, 'new_output'] = new_output
                        count += 1
                        
                        progress_bar.progress((idx + 1) / len(df))
                
                st.session_state.df = df
                status_text.text(f"✅ Tested {count} prompts!")
                st.rerun()
        
        with col3:
            if st.button("✅ Apply All Improvements", use_container_width=True):
                count = 0
                for idx, row in df.iterrows():
                    if pd.notna(row['suggested_prompt']) and row['suggested_prompt'] and \
                       pd.notna(row['new_output']) and row['new_output']:
                        df.at[idx, prompt_col] = row['suggested_prompt']
                        df.at[idx, output_col] = row['new_output']
                        df.at[idx, 'suggested_prompt'] = ""
                        df.at[idx, 'new_output'] = ""
                        count += 1
                
                st.session_state.df = df
                st.success(f"✅ Applied {count} improvements!")
                st.rerun()
        
        # Download section
        st.markdown("---")
        st.header("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"prompt_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Excel download
            excel_data = convert_df_to_excel(df)
            st.download_button(
                label="📥 Download as Excel",
                data=excel_data,
                file_name=f"prompt_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        # Display current DataFrame
        st.markdown("---")
        st.header("📊 Current Data")
        
        # Option to show/hide certain columns
        display_columns = st.multiselect(
            "Select columns to display",
            options=df.columns.tolist(),
            default=[col for col in df.columns if col not in ['suggested_prompt', 'new_output']]
        )
        
        if display_columns:
            st.dataframe(df[display_columns], use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()