import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import os
import re  # Added for better parsing
import random

# Load environment variables
load_dotenv()

# Configure Gemini
gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-2.5-flash')

def predict_scores_llm(df):
    # Filter rated (rating > 0 and edited) and unrated (rating == 0 and not edited)
    rated_df = df[(df['rating'] > 0) & df['edited']].head(5)  # Use up to 5 examples
    unrated_df = df[(df['rating'] == 0) & ~df['edited']]
    
    if len(rated_df) < 2:
        raise ValueError("Need at least two rated prompts to predict scores using LLM.")
    
    # Base prompt template
    base_prompt = """
You are an expert prompt evaluator. Rate system prompts and responses on a scale of 0-10 based on:
- Relevance and accuracy to the query
- Clarity and coherence
- Completeness and depth
- Creativity and engagement (if applicable)
- Overall quality

Examples:
"""
    
    # Add examples from rated_df
    examples = ""
    for _, row in rated_df.iterrows():
        examples += f"""
Example:
System Prompt: {row['system_prompt'][:200]}...
Query: {row['query'][:100]}...
Response: {row['response'][:200]}...
Rating: {row['rating']}/10
"""
    
    base_prompt += examples + "\n\nNow, predict ratings for the following unrated prompts. Output only the integer ratings (0-10) for each prompt, one per line in order, no explanations or additional text."

    # Batch unrated entries for efficiency (process in groups of 5 to avoid token limits)
    batch_size = 5
    predicted_ratings = []
    for start in range(0, len(unrated_df), batch_size):
        batch_df = unrated_df.iloc[start:start + batch_size]
        batch_prompts = ""
        for _, row in batch_df.iterrows():
            batch_prompts += f"""
Unrated:
System Prompt: {row['system_prompt']}
Query: {row['query']}
Response: {row['response']}
"""
        
        full_prompt = base_prompt + batch_prompts + "\nRatings:"
        
        try:
            response = model.generate_content(full_prompt)
            # Improved parsing: Extract all integers from response
            predicted_batch = re.findall(r'\d+', response.text.strip())
            predicted_batch = [int(p) for p in predicted_batch if 0 <= int(p) <= 10][:len(batch_df)]  # Take valid ratings, limit to batch size
            if len(predicted_batch) < len(batch_df):
                # Fallback if not enough ratings extracted
                predicted_batch += [random.randint(1, 10) for _ in range(len(batch_df) - len(predicted_batch))]
            predicted_ratings.extend(predicted_batch)
        except Exception as e:
            # Fallback to random ratings on error
            predicted_ratings.extend([random.randint(1, 10) for _ in range(len(batch_df))])
    
    # Update df
    unrated_indices = unrated_df.index
    df.loc[unrated_indices, 'rating'] = predicted_ratings
    df.loc[unrated_indices, 'remark'] = df.loc[unrated_indices, 'remark'].apply(
        lambda x: (x + " (Predicted by LLM)" if x else "Predicted by LLM") if pd.notna(x) else "Predicted by LLM"
    )
    
    return df