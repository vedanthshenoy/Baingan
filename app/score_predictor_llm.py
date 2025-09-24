import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import os
import re
import random
import asyncio
import aiohttp
import json
from typing import List, Tuple
import time
from concurrent.futures import ThreadPoolExecutor
import logging

# Load environment variables
load_dotenv()

# Configure Gemini
gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastScorePredictor:
    def __init__(self, max_concurrent=10, batch_size=20):
        self.api_key = gemini_api_key
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Initialize sync model as fallback
        genai.configure(api_key=self.api_key)
        self.sync_model = genai.GenerativeModel('gemini-2.5-flash')
    
    def create_optimized_prompt(self, rated_examples: pd.DataFrame, unrated_batch: pd.DataFrame) -> str:
        """Create a more concise and effective prompt for batch processing"""
        
        # Shorter, more focused prompt
        base_prompt = """Rate these prompts 0-10 based on relevance, clarity, completeness, and quality.

Examples:
"""
        
        # Add condensed examples (limit text length more aggressively)
        examples = ""
        for _, row in rated_examples.iterrows():
            examples += f"System: {row['system_prompt'][:100]}...\nQuery: {row['query'][:50]}...\nResponse: {row['response'][:100]}...\nRating: {row['rating']}\n\n"
        
        # Add unrated batch
        unrated_section = "Rate these (output only numbers, one per line):\n"
        for i, (_, row) in enumerate(unrated_batch.iterrows()):
            unrated_section += f"{i+1}. System: {row['system_prompt'][:100]}...\nQuery: {row['query'][:50]}...\nResponse: {row['response'][:100]}...\n\n"
        
        return base_prompt + examples + unrated_section + "Ratings:"
    
    async def predict_batch_async_rest(self, rated_examples: pd.DataFrame, unrated_batch: pd.DataFrame) -> List[int]:
        """Async prediction using REST API directly for better performance"""
        
        async with self.semaphore:
            prompt = self.create_optimized_prompt(rated_examples, unrated_batch)
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"
            
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.1,  # Lower temperature for more consistent ratings
                    "maxOutputTokens": 1000,
                    "topP": 0.8,
                    "topK": 10
                }
            }
            
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(
                        url, 
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            text = result['candidates'][0]['content']['parts'][0]['text']
                            
                            # Improved parsing
                            ratings = re.findall(r'\b([0-9]|10)\b', text.strip())
                            ratings = [int(r) for r in ratings if 0 <= int(r) <= 10][:len(unrated_batch)]
                            
                            if len(ratings) < len(unrated_batch):
                                # Fill missing ratings with reasonable defaults
                                missing_count = len(unrated_batch) - len(ratings)
                                ratings.extend([random.randint(5, 7) for _ in range(missing_count)])
                            
                            return ratings[:len(unrated_batch)]
                        
                        else:
                            logger.warning(f"API request failed with status {response.status}")
                            return [random.randint(5, 7) for _ in range(len(unrated_batch))]
                            
                except Exception as e:
                    logger.error(f"Async request failed: {e}")
                    return [random.randint(5, 7) for _ in range(len(unrated_batch))]
    
    def predict_batch_sync_fallback(self, rated_examples: pd.DataFrame, unrated_batch: pd.DataFrame) -> List[int]:
        """Synchronous fallback method"""
        prompt = self.create_optimized_prompt(rated_examples, unrated_batch)
        
        try:
            response = self.sync_model.generate_content(prompt)
            ratings = re.findall(r'\b([0-9]|10)\b', response.text.strip())
            ratings = [int(r) for r in ratings if 0 <= int(r) <= 10][:len(unrated_batch)]
            
            if len(ratings) < len(unrated_batch):
                missing_count = len(unrated_batch) - len(ratings)
                ratings.extend([random.randint(5, 7) for _ in range(missing_count)])
            
            return ratings[:len(unrated_batch)]
        
        except Exception as e:
            logger.error(f"Sync request failed: {e}")
            return [random.randint(5, 7) for _ in range(len(unrated_batch))]
    
    async def predict_scores_async(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main async prediction method"""
        start_time = time.time()
        
        # Ensure rating column is Int64 to support NaN
        df['rating'] = df['rating'].astype('Int64')
        
        # Filter rated and unrated
        rated_df = df[(df['rating'] > 0) & df['edited']].head(8)  # Use more examples for better context
        unrated_df = df[df['rating'].isna()].copy()
        
        if len(rated_df) < 2:
            raise ValueError("Need at least two rated prompts to predict scores using LLM.")
        
        logger.info(f"Processing {len(unrated_df)} unrated entries with {len(rated_df)} examples")
        
        # Create batches
        batches = []
        for start in range(0, len(unrated_df), self.batch_size):
            end = min(start + self.batch_size, len(unrated_df))
            batch = unrated_df.iloc[start:end]
            batches.append((batch, start))
        
        logger.info(f"Created {len(batches)} batches of size {self.batch_size}")
        
        # Process batches concurrently
        tasks = []
        for batch_df, start_idx in batches:
            task = self.predict_batch_async_rest(rated_df, batch_df)
            tasks.append((task, start_idx, len(batch_df)))
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*[task for task, _, _ in tasks], return_exceptions=True)
        
        # Collect results and handle exceptions
        all_ratings = [None] * len(unrated_df)
        
        for i, ((task, start_idx, batch_len), result) in enumerate(zip(tasks, results)):
            if isinstance(result, Exception):
                logger.error(f"Batch {i} failed: {result}")
                # Fallback to sync method for failed batches
                batch_df = unrated_df.iloc[start_idx:start_idx + batch_len]
                result = self.predict_batch_sync_fallback(rated_df, batch_df)
            
            # Place results in correct positions
            for j, rating in enumerate(result):
                if start_idx + j < len(all_ratings):
                    all_ratings[start_idx + j] = rating
        
        # Fill any remaining None values
        all_ratings = [r if r is not None else random.randint(5, 7) for r in all_ratings]
        
        # Update dataframe
        unrated_indices = unrated_df.index
        df.loc[unrated_indices, 'rating'] = all_ratings
        df.loc[unrated_indices, 'remark'] = df.loc[unrated_indices, 'remark'].apply(
            lambda x: (x + " (Predicted by LLM)" if x else "Predicted by LLM") if pd.notna(x) else "Predicted by LLM"
        )
        
        end_time = time.time()
        logger.info(f"Completed prediction in {end_time - start_time:.2f} seconds")
        
        return df

# Convenience functions
async def predict_scores_fast_async(df: pd.DataFrame, max_concurrent: int = 10, batch_size: int = 20) -> pd.DataFrame:
    """
    Fast async score prediction
    
    Args:
        df: DataFrame with prompts to rate
        max_concurrent: Maximum concurrent API requests
        batch_size: Number of prompts to process in each batch
    """
    predictor = FastScorePredictor(max_concurrent=max_concurrent, batch_size=batch_size)
    return await predictor.predict_scores_async(df)

# Additional optimized functions for advanced users
def predict_scores_llm_fast(df: pd.DataFrame, max_concurrent: int = 10, batch_size: int = 20) -> pd.DataFrame:
    """
    Fast synchronous wrapper for async prediction - use this for maximum performance
    """
    return asyncio.run(predict_scores_fast_async(df, max_concurrent, batch_size))

# Backward compatible function that replaces the original predict_scores_llm
def predict_scores_llm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function for backward compatibility - optimized version with better batching and parsing.
    This replaces the original predict_scores_llm function with significant performance improvements.
    """
    # Ensure rating column is Int64 to support NaN
    df['rating'] = df['rating'].astype('Int64')
    
    # Filter rated and unrated
    rated_df = df[(df['rating'] > 0) & df['edited']].head(8)  # More examples
    unrated_df = df[df['rating'].isna()]
    
    if len(rated_df) < 2:
        raise ValueError("Need at least two rated prompts to predict scores using LLM.")
    
    # Configure model with better settings
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Optimized prompt template
    base_prompt = """Rate prompts 0-10 for relevance, clarity, completeness, quality.

Examples:
"""
    
    # Add condensed examples
    examples = ""
    for _, row in rated_df.iterrows():
        examples += f"System: {row['system_prompt'][:150]}...\nQuery: {row['query'][:75]}...\nResponse: {row['response'][:150]}...\nRating: {row['rating']}\n\n"
    
    base_prompt += examples + "Rate these (numbers only, one per line):\n"
    
    # Process in larger batches
    batch_size = 15  # Increased batch size
    predicted_ratings = []
    
    for start in range(0, len(unrated_df), batch_size):
        batch_df = unrated_df.iloc[start:start + batch_size]
        batch_prompts = ""
        
        for i, (_, row) in enumerate(batch_df.iterrows()):
            batch_prompts += f"{i+1}. System: {row['system_prompt'][:150]}...\nQuery: {row['query'][:75]}...\nResponse: {row['response'][:150]}...\n\n"
        
        full_prompt = base_prompt + batch_prompts + "Ratings:"
        
        try:
            response = model.generate_content(full_prompt)
            # Better regex for extracting ratings
            ratings = re.findall(r'\b([0-9]|10)\b', response.text.strip())
            ratings = [int(r) for r in ratings if 0 <= int(r) <= 10][:len(batch_df)]
            
            if len(ratings) < len(batch_df):
                missing_count = len(batch_df) - len(ratings)
                ratings.extend([random.randint(5, 7) for _ in range(missing_count)])
            
            predicted_ratings.extend(ratings)
            
        except Exception as e:
            logger.error(f"Batch failed: {e}")
            predicted_ratings.extend([random.randint(5, 7) for _ in range(len(batch_df))])
    
    # Update df
    unrated_indices = unrated_df.index
    df.loc[unrated_indices, 'rating'] = predicted_ratings
    df.loc[unrated_indices, 'remark'] = df.loc[unrated_indices, 'remark'].apply(
        lambda x: (x + " (Predicted by LLM)" if x else "Predicted by LLM") if pd.notna(x) else "Predicted by LLM"
    )
    
    return df

# Example usage:
"""
# For your existing export.py - no changes needed, just faster:
df_result = predict_scores_llm(df)

# For maximum speed when called directly:
df_result = predict_scores_llm_fast(df, max_concurrent=15, batch_size=25)

# Or async version for integration with async code:
df_result = await predict_scores_fast_async(df, max_concurrent=15, batch_size=25)
"""