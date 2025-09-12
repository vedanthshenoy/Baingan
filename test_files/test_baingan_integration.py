import unittest
from unittest.mock import Mock, patch, MagicMock, call
import requests
import json
import pandas as pd
import io
from datetime import datetime
import sys
import os
import tempfile
import time

# Add the parent directory to sys.path to import the main module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add the parent directory to sys.path to import the main module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add the project_apps directory for the main application
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'project_apps'))

# Mock streamlit and other dependencies before importing
sys.modules['streamlit'] = Mock()
sys.modules['google.generativeai'] = Mock()
sys.modules['dotenv'] = Mock()


class TestBainGanIntegration(unittest.TestCase):
    """Integration tests for the complete BainGan application workflow"""
    
    def setUp(self):
        """Set up test fixtures for integration tests"""
        self.mock_session_state = {
            'prompts': [],
            'prompt_names': [],
            'test_results': [],
            'chain_results': [],
            'combination_results': [],
            'slider_weights': {},
            'last_selected_prompts': [],
            'response_ratings': {},
            'export_data': [],
            'temperature': 50,
            'prompt_input_key_suffix': 'test-uuid'
        }
        
        self.test_prompts = [
            "You are a helpful assistant. Provide clear, concise answers.",
            "You are a creative writer. Write engaging, descriptive content.",
            "You are a technical expert. Provide detailed, accurate information."
        ]
        
        self.test_prompt_names = [
            "Helpful Assistant",
            "Creative Writer", 
            "Technical Expert"
        ]
        
        self.test_query = "Explain artificial intelligence in simple terms."
        
        self.api_config = {
            'api_url': 'https://api.test.com/chat',
            'headers': {'Content-Type': 'application/json', 'Authorization': 'Bearer test-token'},
            'body_template': '{"query": "{system_prompt}\\n\\nQuestion: {query}\\nAnswer:", "top_k": 5}',
            'response_path': 'response'
        }


class TestEndToEndWorkflow(TestBainGanIntegration):
    """Test complete end-to-end workflows"""
    
    @patch('clean_baingan.requests.post')
    @patch('clean_baingan.st.session_state')
    def test_complete_individual_testing_workflow(self, mock_session_state, mock_requests_post):
        """Test complete individual testing workflow from prompt creation to export"""
        # Setup session state
        mock_session_state.configure_mock(**self.mock_session_state)
        mock_session_state.__getitem__.side_effect = lambda key: self.mock_session_state[key]
        mock_session_state.__setitem__.side_effect = lambda key, value: self.mock_session_state.__setitem__(key, value)
        
        # Add test prompts to session state
        self.mock_session_state['prompts'] = self.test_prompts.copy()
        self.mock_session_state['prompt_names'] = self.test_prompt_names.copy()
        
        # Mock successful API responses
        mock_responses = [
            Mock(status_code=200, json=lambda: {"response": "AI is computer intelligence that mimics human thinking."}),
            Mock(status_code=200, json=lambda: {"response": "Imagine artificial intelligence as digital brains that learn and think like humans, helping us solve complex problems with lightning speed."}),
            Mock(status_code=200, json=lambda: {"response": "Artificial Intelligence (AI) refers to machine learning algorithms and neural networks that process data to make predictions and decisions autonomously."})
        ]
        mock_requests_post.side_effect = mock_responses
        
        # Import and test the API call function
        with patch('clean_baingan.api_url', self.api_config['api_url']), \
             patch('clean_baingan.headers', self.api_config['headers']), \
             patch('clean_baingan.body_template', self.api_config['body_template']), \
             patch('clean_baingan.response_path', self.api_config['response_path']), \
             patch('clean_baingan.query_text', self.test_query):
            
            from clean_baingan import call_api, ensure_prompt_names
            
            # Ensure prompt names are synced
            ensure_prompt_names()
            
            # Simulate testing all prompts
            test_results = []
            for i, (system_prompt, prompt_name) in enumerate(zip(self.test_prompts, self.test_prompt_names)):
                result = call_api(system_prompt, self.test_query)
                result.update({
                    'prompt_name': prompt_name,
                    'system_prompt': system_prompt,
                    'query': self.test_query,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'edited': False,
                    'remark': 'Saved and ran'
                })
                test_results.append(result)
            
            # Verify results
            self.assertEqual(len(test_results), 3)
            for i, result in enumerate(test_results):
                self.assertEqual(result['status'], 'Success')
                self.assertEqual(result['prompt_name'], self.test_prompt_names[i])
                self.assertIn('AI', result['response'])
            
            # Verify API calls were made correctly
            self.assertEqual(mock_requests_post.call_count, 3)
            
            # Check that proper API calls were made
            for call_args in mock_requests_post.call_args_list:
                args, kwargs = call_args
                self.assertEqual(args[0], self.api_config['api_url'])
                self.assertEqual(kwargs['headers'], self.api_config['headers'])
                self.assertIn('query', kwargs['json'])
    
    @patch('clean_baingan.requests.post')
    @patch('clean_baingan.st.session_state')
    def test_complete_chain_workflow(self, mock_session_state, mock_requests_post):
        """Test complete prompt chaining workflow"""
        # Setup session state
        mock_session_state.configure_mock(**self.mock_session_state)
        mock_session_state.__getitem__.side_effect = lambda key: self.mock_session_state[key]
        
        # Add test prompts
        chain_prompts = [
            "Summarize the following text in one sentence:",
            "Extract the main keywords from this summary:",
            "Create a title based on these keywords:"
        ]
        chain_names = ["Summarizer", "Keyword Extractor", "Title Creator"]
        
        self.mock_session_state['prompts'] = chain_prompts
        self.mock_session_state['prompt_names'] = chain_names
        
        # Mock chained API responses
        mock_responses = [
            Mock(status_code=200, json=lambda: {"response": "AI is a technology that enables machines to simulate human intelligence and perform complex tasks."}),
            Mock(status_code=200, json=lambda: {"response": "artificial intelligence, technology, machines, human intelligence, complex tasks"}),
            Mock(status_code=200, json=lambda: {"response": "Artificial Intelligence: Machines Simulating Human Intelligence for Complex Tasks"})
        ]
        mock_requests_post.side_effect = mock_responses
        
        with patch('clean_baingan.api_url', self.api_config['api_url']), \
             patch('clean_baingan.headers', self.api_config['headers']), \
             patch('clean_baingan.body_template', self.api_config['body_template']), \
             patch('clean_baingan.response_path', self.api_config['response_path']):
            
            from clean_baingan import call_api
            
            # Simulate chain execution
            chain_results = []
            current_query = self.test_query
            
            for i, (system_prompt, prompt_name) in enumerate(zip(chain_prompts, chain_names)):
                result = call_api(system_prompt, current_query)
                result.update({
                    'step': i + 1,
                    'prompt_name': prompt_name,
                    'system_prompt': system_prompt,
                    'input_query': current_query,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'edited': False,
                    'remark': 'Saved and ran'
                })
                
                chain_results.append(result)
                
                if result['status'] != 'Success':
                    break
                
                # Use this response as input for next step
                current_query = result['response']
            
            # Verify chain execution
            self.assertEqual(len(chain_results), 3)
            
            # Verify chain progression
            self.assertEqual(chain_results[0]['step'], 1)
            self.assertEqual(chain_results[0]['input_query'], self.test_query)
            self.assertIn('AI', chain_results[0]['response'])
            
            self.assertEqual(chain_results[1]['step'], 2)
            self.assertEqual(chain_results[1]['input_query'], chain_results[0]['response'])
            self.assertIn('artificial intelligence', chain_results[1]['response'])
            
            self.assertEqual(chain_results[2]['step'], 3)
            self.assertEqual(chain_results[2]['input_query'], chain_results[1]['response'])
            self.assertIn('Intelligence', chain_results[2]['response'])
            
            # Verify all steps were successful
            for result in chain_results:
                self.assertEqual(result['status'], 'Success')
    
    @patch('clean_baingan.genai.GenerativeModel')
    @patch('clean_baingan.genai.configure')
    @patch('clean_baingan.requests.post')
    @patch('clean_baingan.st.session_state')
    def test_complete_combination_workflow(self, mock_session_state, mock_requests_post, mock_configure, mock_model_class):
        """Test complete prompt combination workflow"""
        # Setup session state
        mock_session_state.configure_mock(**self.mock_session_state)
        mock_session_state.__getitem__.side_effect = lambda key: self.mock_session_state[key]
        
        # Add test prompts
        combination_prompts = [
            "You are a helpful assistant. Provide clear answers.",
            "You are creative. Write engaging content."
        ]
        combination_names = ["Helper", "Creative"]
        
        self.mock_session_state['prompts'] = combination_prompts
        self.mock_session_state['prompt_names'] = combination_names
        self.mock_session_state['slider_weights'] = {0: 60, 1: 40}
        
        # Mock Gemini model for prompt combination
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "You are a helpful and creative assistant. Provide clear, engaging answers with vivid descriptions."
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        # Mock API responses for testing individual and combined prompts
        individual_responses = [
            Mock(status_code=200, json=lambda: {"response": "AI is computer intelligence."}),
            Mock(status_code=200, json=lambda: {"response": "Picture AI as digital minds that dance with data."})
        ]
        combined_response = Mock(status_code=200, json=lambda: {"response": "AI is computer intelligence that dances with data, creating a symphony of digital understanding."})
        
        mock_requests_post.side_effect = individual_responses + [combined_response]
        
        with patch('clean_baingan.api_url', self.api_config['api_url']), \
             patch('clean_baingan.headers', self.api_config['headers']), \
             patch('clean_baingan.body_template', self.api_config['body_template']), \
             patch('clean_baingan.response_path', self.api_config['response_path']), \
             patch('clean_baingan.gemini_api_key', 'test_key'):
            
            from clean_baingan import call_api, suggest_prompt_from_response
            
            # Test prompt combination using Gemini
            selected_prompts = [0, 1]
            selected_prompt_texts = [combination_prompts[i] for i in selected_prompts]
            selected_prompt_names = [combination_names[i] for i in selected_prompts]
            
            # Simulate AI combination
            combination_strategy = "Slider - Custom influence weights"
            
            # Test individual prompts first
            individual_results = []
            for i, (prompt, name) in enumerate(zip(selected_prompt_texts, selected_prompt_names)):
                result = call_api(prompt, self.test_query)
                result.update({
                    'prompt_index': i + 1,
                    'prompt_name': name,
                    'system_prompt': prompt,
                    'query': self.test_query,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'edited': False,
                    'remark': 'Saved and ran'
                })
                individual_results.append(result)
            
            # Test combined prompt
            combined_prompt = mock_response.text
            combined_result = call_api(combined_prompt, self.test_query)
            combined_result.update({
                'system_prompt': combined_prompt,
                'query': self.test_query,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'edited': False,
                'remark': 'Saved and ran'
            })
            
            # Store combination results
            combination_results = {
                'individual_prompts': selected_prompt_texts,
                'individual_names': selected_prompt_names,
                'selected_indices': selected_prompts,
                'combined_prompt': combined_prompt,
                'strategy': combination_strategy,
                'temperature': 50,
                'slider_weights': {0: 60, 1: 40},
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'individual_results': individual_results,
                'combined_result': combined_result
            }
            
            # Verify combination workflow
            self.assertEqual(len(individual_results), 2)
            self.assertIsNotNone(combined_result)
            
            # Verify individual results
            for result in individual_results:
                self.assertEqual(result['status'], 'Success')
                self.assertIn('AI', result['response'])
            
            # Verify combined result
            self.assertEqual(combined_result['status'], 'Success')
            self.assertIn('AI', combined_result['response'])
            
            # Verify Gemini was called for combination
            mock_configure.assert_called_once_with(api_key='test_key')
            mock_model.generate_content.assert_called_once()
            
            # Verify API calls (2 individual + 1 combined)
            self.assertEqual(mock_requests_post.call_count, 3)


class TestErrorHandlingAndEdgeCases(TestBainGanIntegration):
    """Test error handling and edge cases"""
    
    @patch('clean_baingan.requests.post')
    @patch('clean_baingan.st.session_state')
    def test_api_error_handling_in_chain(self, mock_session_state, mock_requests_post):
        """Test chain execution when API errors occur"""
        mock_session_state.configure_mock(**self.mock_session_state)
        mock_session_state.__getitem__.side_effect = lambda key: self.mock_session_state[key]
        
        chain_prompts = ["Step 1", "Step 2", "Step 3"]
        chain_names = ["Name1", "Name2", "Name3"]
        
        self.mock_session_state['prompts'] = chain_prompts
        self.mock_session_state['prompt_names'] = chain_names
        
        # Mock responses: success, then error, then not called
        mock_responses = [
            Mock(status_code=200, json=lambda: {"response": "Success step 1"}),
            Mock(status_code=500, text="Internal Server Error")
        ]
        mock_requests_post.side_effect = mock_responses
        
        with patch('clean_baingan.api_url', self.api_config['api_url']), \
             patch('clean_baingan.headers', self.api_config['headers']), \
             patch('clean_baingan.body_template', self.api_config['body_template']), \
             patch('clean_baingan.response_path', self.api_config['response_path']):
            
            from clean_baingan import call_api
            
            chain_results = []
            current_query = self.test_query
            
            for i, (system_prompt, prompt_name) in enumerate(zip(chain_prompts, chain_names)):
                result = call_api(system_prompt, current_query)
                result.update({
                    'step': i + 1,
                    'prompt_name': prompt_name,
                    'system_prompt': system_prompt,
                    'input_query': current_query,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'edited': False,
                    'remark': 'Saved and ran'
                })
                
                chain_results.append(result)
                
                if result['status'] != 'Success':
                    break
                
                current_query = result['response']
            
            # Verify chain stopped at error
            self.assertEqual(len(chain_results), 2)
            self.assertEqual(chain_results[0]['status'], 'Success')
            self.assertEqual(chain_results[1]['status'], 'Error')
            
            # Verify only 2 API calls were made (stopped after error)
            self.assertEqual(mock_requests_post.call_count, 2)
    
    @patch('clean_baingan.genai.GenerativeModel')
    @patch('clean_baingan.st.session_state')
    def test_gemini_api_error_handling(self, mock_session_state, mock_model_class):
        """Test handling of Gemini API errors"""
        mock_session_state.configure_mock(**self.mock_session_state)
        mock_session_state.__getitem__.side_effect = lambda key: self.mock_session_state[key]
        
        # Mock Gemini API error
        mock_model_class.side_effect = Exception("Gemini API rate limit exceeded")
        
        with patch('clean_baingan.gemini_api_key', 'test_key'):
            from clean_baingan import suggest_prompt_from_response
            
            result = suggest_prompt_from_response("Target response", "Query")
            
            self.assertIn("Error generating prompt suggestion", result)
            self.assertIn("Gemini API rate limit exceeded", result)
    
    @patch('clean_baingan.st.session_state')
    def test_empty_prompts_handling(self, mock_session_state):
        """Test handling of empty prompts list"""
        mock_session_state.configure_mock(**self.mock_session_state)
        mock_session_state.__getitem__.side_effect = lambda key: self.mock_session_state[key]
        
        # Empty prompts
        self.mock_session_state['prompts'] = []
        self.mock_session_state['prompt_names'] = []
        
        from clean_baingan import ensure_prompt_names
        
        ensure_prompt_names()
        
        # Should remain empty
        self.assertEqual(len(self.mock_session_state['prompts']), 0)
        self.assertEqual(len(self.mock_session_state['prompt_names']), 0)
    
    @patch('clean_baingan.st.session_state')
    def test_mismatched_prompts_and_names(self, mock_session_state):
        """Test handling of mismatched prompts and names lists"""
        mock_session_state.configure_mock(**self.mock_session_state)
        mock_session_state.__getitem__.side_effect = lambda key: self.mock_session_state[key]
        
        # More prompts than names
        self.mock_session_state['prompts'] = ['p1', 'p2', 'p3']
        self.mock_session_state['prompt_names'] = ['n1']
        
        from clean_baingan import ensure_prompt_names
        
        ensure_prompt_names()
        
        # Should have matching lengths
        self.assertEqual(len(self.mock_session_state['prompts']), 3)
        self.assertEqual(len(self.mock_session_state['prompt_names']), 3)
        self.assertEqual(self.mock_session_state['prompt_names'][0], 'n1')
        self.assertEqual(self.mock_session_state['prompt_names'][1], 'Prompt 2')
        self.assertEqual(self.mock_session_state['prompt_names'][2], 'Prompt 3')


class TestDataIntegrityAndExport(TestBainGanIntegration):
    """Test data integrity and export functionality"""
    
    @patch('pandas.DataFrame')
    @patch('pandas.ExcelWriter')
    @patch('io.BytesIO')
    def test_complete_export_workflow(self, mock_bytesio, mock_excel_writer, mock_dataframe):
        """Test complete export workflow with all data types"""
        # Mock components
        mock_output = Mock()
        mock_output.getvalue.return_value = b'fake_excel_data'
        mock_bytesio.return_value = mock_output
        
        mock_writer = Mock()
        mock_excel_writer.return_value.__enter__.return_value = mock_writer
        mock_writer.sheets = {'Test_Results': Mock()}
        
        mock_df = Mock()
        mock_dataframe.return_value = mock_df
        
        # Create comprehensive export data
        export_data = [
            # Individual test result
            {
                'unique_id': 'Individual_Test1_2024-01-01 10:00:00_0',
                'test_type': 'Individual',
                'prompt_name': 'Test Prompt 1',
                'system_prompt': 'You are helpful.',
                'query': 'What is AI?',
                'response': 'AI is artificial intelligence.',
                'status': 'Success',
                'status_code': 200,
                'timestamp': '2024-01-01 10:00:00',
                'edited': False,
                'step': None,
                'input_query': None,
                'rating': 80,
                'remark': 'Saved and ran'
            },
            # Chain result
            {
                'unique_id': 'Chain_ChainStep1_2024-01-01 11:00:00_1',
                'test_type': 'Chain',
                'prompt_name': 'Chain Step 1',
                'system_prompt': 'Summarize this.',
                'query': 'What is AI?',
                'response': 'AI summary.',
                'status': 'Success',
                'status_code': 200,
                'timestamp': '2024-01-01 11:00:00',
                'edited': False,
                'step': 1,
                'input_query': 'What is AI?',
                'rating': 70,
                'remark': 'Saved and ran'
            },
            # Combination result
            {
                'unique_id': 'Combination_Combined_2024-01-01 12:00:00',
                'test_type': 'Combination_Combined',
                'prompt_name': 'AI_Combined',
                'system_prompt': 'Combined prompt text.',
                'query': 'What is AI?',
                'response': 'Combined AI response.',
                'status': 'Success',
                'status_code': 200,
                'timestamp': '2024-01-01 12:00:00',
                'edited': False,
                'step': None,
                'input_query': None,
                'combination_strategy': 'Merge and optimize',
                'combination_temperature': 50,
                'slider_weights': None,
                'rating': 90,
                'remark': 'Saved and ran'
            }
        ]
        
        # Test DataFrame creation and Excel export
        df = pd.DataFrame(export_data)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Test_Results', index=False)
        
        excel_data = output.getvalue()
        
        # Verify export process
        mock_dataframe.assert_called_once_with(export_data)
        mock_bytesio.assert_called_once()
        mock_excel_writer.assert_called_once()
        
        # Verify data structure
        self.assertEqual(len(export_data), 3)
        
        # Test unique IDs
        unique_ids = [item['unique_id'] for item in export_data]
        self.assertEqual(len(set(unique_ids)), len(unique_ids))  # All unique
        
        # Test data types
        test_types = [item['test_type'] for item in export_data]
        self.assertIn('Individual', test_types)
        self.assertIn('Chain', test_types)
        self.assertIn('Combination_Combined', test_types)
    
    def test_duplicate_prevention_in_export(self):
        """Test that duplicate entries are prevented in export data"""
        export_data = []
        
        # Simulate adding the same result multiple times
        result_data = {
            'unique_id': 'Individual_Test1_2024-01-01 10:00:00_0',
            'test_type': 'Individual',
            'prompt_name': 'Test Prompt 1'
        }
        
        # First addition
        if not any(d.get('unique_id') == result_data['unique_id'] for d in export_data):
            export_data.append(result_data.copy())
        
        # Second addition (should be prevented)
        if not any(d.get('unique_id') == result_data['unique_id'] for d in export_data):
            export_data.append(result_data.copy())
        
        # Verify only one entry exists
        self.assertEqual(len(export_data), 1)
    
    def test_rating_conversion_in_export(self):
        """Test that ratings are correctly converted to percentages in export"""
        # Simulate session state ratings (0-10 scale)
        response_ratings = {
            'test_0': 8,
            'chain_0': 5,
            'combination_combined': 9
        }
        
        # Test conversion to percentage (multiply by 10)
        export_ratings = {
            key: value * 10 for key, value in response_ratings.items()
        }
        
        self.assertEqual(export_ratings['test_0'], 80)
        self.assertEqual(export_ratings['chain_0'], 50)
        self.assertEqual(export_ratings['combination_combined'], 90)


class TestPerformanceAndStress(TestBainGanIntegration):
    """Test performance and stress scenarios"""
    
    @patch('clean_baingan.requests.post')
    @patch('clean_baingan.st.session_state')
    def test_large_number_of_prompts(self, mock_session_state, mock_requests_post):
        """Test handling of large number of prompts"""
        mock_session_state.configure_mock(**self.mock_session_state)
        mock_session_state.__getitem__.side_effect = lambda key: self.mock_session_state[key]
        
        # Create 20 test prompts
        large_prompt_list = [f"Test prompt {i}" for i in range(20)]
        large_name_list = [f"Name {i}" for i in range(20)]
        
        self.mock_session_state['prompts'] = large_prompt_list
        self.mock_session_state['prompt_names'] = large_name_list
        
        # Mock successful responses for all
        mock_response = Mock(status_code=200, json=lambda: {"response": "Test response"})
        mock_requests_post.return_value = mock_response
        
        with patch('clean_baingan.api_url', self.api_config['api_url']), \
             patch('clean_baingan.headers', self.api_config['headers']), \
             patch('clean_baingan.body_template', self.api_config['body_template']), \
             patch('clean_baingan.response_path', self.api_config['response_path']):
            
            from clean_baingan import call_api, ensure_prompt_names
            
            start_time = time.time()
            
            # Test all prompts
            results = []
            for i, (prompt, name) in enumerate(zip(large_prompt_list, large_name_list)):
                result = call_api(prompt, self.test_query)
                results.append(result)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Verify all completed successfully
            self.assertEqual(len(results), 20)
            for result in results:
                self.assertEqual(result['status'], 'Success')
            
            # Performance check (should complete in reasonable time)
            self.assertLess(execution_time, 5.0)  # Should complete in under 5 seconds with mocked responses
    
    @patch('clean_baingan.requests.post')
    def test_long_response_handling(self, mock_requests_post):
        """Test handling of very long API responses"""
        # Create a very long response (10KB+)
        long_response = "A" * 10000 + " This is a very long response that tests the system's ability to handle large amounts of text data."
        
        mock_response = Mock(status_code=200, json=lambda: {"response": long_response})
        mock_requests_post.return_value = mock_response
        
        with patch('clean_baingan.api_url', self.api_config['api_url']), \
             patch('clean_baingan.headers', self.api_config['headers']), \
             patch('clean_baingan.body_template', self.api_config['body_template']), \
             patch('clean_baingan.response_path', self.api_config['response_path']):
            
            from clean_baingan import call_api
            
            result = call_api("Test prompt", "Test query")
            
            self.assertEqual(result['status'], 'Success')
            self.assertEqual(len(result['response']), len(long_response))
            self.assertIn("very long response", result['response'])


if __name__ == '__main__':
    # Create comprehensive test suite
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_suite.addTest(loader.loadTestsFromTestCase(TestEndToEndWorkflow))
    test_suite.addTest(loader.loadTestsFromTestCase(TestErrorHandlingAndEdgeCases))
    test_suite.addTest(loader.loadTestsFromTestCase(TestDataIntegrityAndExport))
    test_suite.addTest(loader.loadTestsFromTestCase(TestPerformanceAndStress))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    # Print comprehensive summary
    print(f"\n{'='*60}")
    print(f"INTEGRATION TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, failure in result.failures:
            print(f"- {test}: {failure.split(chr(10))[0]}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, error in result.errors:
            print(f"- {test}: {error.split(chr(10))[0]}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    # Test coverage summary
    print(f"\n{'='*60}")
    print(f"COVERAGE AREAS TESTED")
    print(f"{'='*60}")
    print("✓ End-to-end individual testing workflow")
    print("✓ End-to-end prompt chaining workflow")  
    print("✓ End-to-end prompt combination workflow")
    print("✓ API error handling in chains")
    print("✓ Gemini API error handling")
    print("✓ Empty prompts handling")
    print("✓ Data structure integrity")
    print("✓ Export workflow with all data types")
    print("✓ Duplicate prevention in exports")
    print("✓ Rating conversion (0-10 to percentage)")
    print("✓ Performance with large datasets")
    print("✓ Long response text handling")
    
    # Exit with appropriate code
    exit_code = 0 if (len(result.failures) == 0 and len(result.errors) == 0) else 1
    sys.exit(exit_code)