import unittest
from unittest.mock import Mock, patch
import sys
import os

# Ensure project_apps is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'project_apps'))

# ---- Minimal Streamlit Mock ----
class MockSessionState(dict):
    """Dictionary-like session state for tests."""
    def __getattr__(self, name):
        return self.get(name, None)
    def __setattr__(self, name, value):
        self[name] = value

mock_st = Mock()
mock_st.session_state = MockSessionState()
sys.modules['streamlit'] = mock_st

# Mock external libs
sys.modules['google.generativeai'] = Mock()
sys.modules['dotenv'] = Mock()

# Import after mocking
from clean_baingan import ensure_prompt_names, call_api, suggest_prompt_from_response


class TestBainGanUtilities(unittest.TestCase):
    """Test utility functions from Baingan app"""

    def setUp(self):
        self.mock_session_state = MockSessionState({
            'prompts': [],
            'prompt_names': [],
            'test_results': [],
            'chain_results': [],
            'combination_results': [],
            'slider_weights': {},
            'last_selected_prompts': [],
            'response_ratings': {}
        })

    def test_ensure_prompt_names_empty_lists(self):
        with patch('clean_baingan.st.session_state', self.mock_session_state):
            ensure_prompt_names()
            self.assertEqual(len(self.mock_session_state['prompt_names']), 0)
            self.assertEqual(len(self.mock_session_state['prompts']), 0)

    def test_ensure_prompt_names_with_prompts(self):
        self.mock_session_state['prompts'] = ['prompt1', 'prompt2', 'prompt3']
        with patch('clean_baingan.st.session_state', self.mock_session_state):
            ensure_prompt_names()
            self.assertEqual(len(self.mock_session_state['prompt_names']), 3)
            self.assertEqual(self.mock_session_state['prompt_names'][0], 'Prompt 1')

    def test_ensure_prompt_names_with_extra_names(self):
        self.mock_session_state['prompts'] = ['prompt1']
        self.mock_session_state['prompt_names'] = ['Name1', 'Name2', 'Name3']
        with patch('clean_baingan.st.session_state', self.mock_session_state):
            ensure_prompt_names()
            self.assertEqual(len(self.mock_session_state['prompt_names']), 1)
            self.assertEqual(self.mock_session_state['prompt_names'][0], 'Name1')


class TestAPICall(unittest.TestCase):
    """Test API call functionality"""

    def setUp(self):
        self.system_prompt = "You are a helpful assistant."
        self.query = "What is AI?"
        self.api_url = "https://api.example.com/chat"
        self.headers = {"Content-Type": "application/json"}
        # Safe template that won't crash when formatted
        self.body_template = '{"system_prompt": "{system_prompt}", "query": "{query}"}'
        self.response_path = "response"

    @patch('clean_baingan.requests.post')
    def test_call_api_success(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "AI is artificial intelligence."}
        mock_post.return_value = mock_response

        result = call_api(
            self.system_prompt, self.query,
            api_url=self.api_url,
            headers=self.headers,
            body_template=self.body_template,
            response_path=self.response_path
        )

        self.assertEqual(result['status'], 'Success')
        self.assertEqual(result['response'], 'AI is artificial intelligence.')
        self.assertEqual(result['status_code'], 200)

    @patch('clean_baingan.requests.post')
    def test_call_api_http_error(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.return_value = mock_response

        result = call_api(
            self.system_prompt, self.query,
            api_url=self.api_url,
            headers=self.headers,
            body_template=self.body_template,
            response_path=self.response_path
        )

        self.assertEqual(result['status'], 'Error')
        self.assertEqual(result['status_code'], 400)
        self.assertIn('Bad Request', result['response'])

    @patch('clean_baingan.requests.post')
    def test_call_api_timeout_exception(self, mock_post):
        mock_post.side_effect = Exception("Connection timeout")

        result = call_api(
            self.system_prompt, self.query,
            api_url=self.api_url,
            headers=self.headers,
            body_template=self.body_template,
            response_path=self.response_path
        )

        self.assertEqual(result['status'], 'Unknown Error')
        self.assertEqual(result['status_code'], 'N/A')
        self.assertIn('Connection timeout', result['response'])

    @patch('clean_baingan.requests.post')
    def test_call_api_nested_response_path(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {"message": "Nested response content"}
        }
        mock_post.return_value = mock_response

        result = call_api(
            self.system_prompt, self.query,
            api_url=self.api_url,
            headers=self.headers,
            body_template=self.body_template,
            response_path='data.message'
        )

        self.assertEqual(result['status'], 'Success')
        self.assertEqual(result['response'], 'Nested response content')


class TestPromptSuggestion(unittest.TestCase):
    """Test AI prompt suggestion functionality"""

    def setUp(self):
        self.target_response = "This is a detailed, helpful response."
        self.query = "How do I learn programming?"
        self.gemini_api_key = "test_api_key"

    @patch('clean_baingan.genai.GenerativeModel')
    @patch('clean_baingan.genai.configure')
    def test_suggest_prompt_from_response_success(self, mock_configure, mock_model_class):
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "You are a programming tutor."
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        mock_st.session_state['temperature'] = 50

        result = suggest_prompt_from_response(
            self.target_response,
            self.query,
            gemini_api_key=self.gemini_api_key
        )

        self.assertEqual(result, "You are a programming tutor.")
        mock_configure.assert_called_once_with(api_key=self.gemini_api_key)
        mock_model.generate_content.assert_called_once()

    def test_suggest_prompt_no_api_key(self):
        result = suggest_prompt_from_response(
            self.target_response, self.query, gemini_api_key=None
        )
        self.assertEqual(result, "Gemini API key required for prompt suggestion")

    @patch('clean_baingan.genai.GenerativeModel')
    @patch('clean_baingan.genai.configure')
    def test_suggest_prompt_api_error(self, mock_configure, mock_model_class):
        mock_model_class.side_effect = Exception("API rate limit exceeded")
        mock_st.session_state['temperature'] = 50

        result = suggest_prompt_from_response(
            self.target_response,
            self.query,
            gemini_api_key=self.gemini_api_key
        )

        self.assertIn("Error generating prompt suggestion", result)
        self.assertIn("API rate limit exceeded", result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
