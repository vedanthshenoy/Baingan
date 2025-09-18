import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import Sidebar from './components/Sidebar';
import PromptManager from './components/PromptManager';
import IndividualMode from './components/IndividualMode';
import ChainingMode from './components/ChainingMode';
import CombinationMode from './components/CombinationMode';
import ExportSection from './components/ExportSection';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const App = () => {
  // State management
  const [apiConfig, setApiConfig] = useState({
    api_url: '',
    auth_type: 'None',
    bearer_token: '',
    api_key: '',
    key_header: 'X-API-Key',
    custom_headers: {},
    body_template: '{"query": "{system_prompt}\\n\\nQuestion: {query}\\nAnswer:", "top_k": 5}',
    response_path: 'response'
  });
  
  const [query, setQuery] = useState('');
  const [testMode, setTestMode] = useState('Individual Testing');
  const [prompts, setPrompts] = useState([]);
  const [promptNames, setPromptNames] = useState([]);
  const [testResults, setTestResults] = useState([]);
  const [chainResults, setChainResults] = useState([]);
  const [combinationResults, setCombinationResults] = useState({});
  const [loading, setLoading] = useState(false);

  // Load initial data
  useEffect(() => {
    loadPrompts();
  }, []);

  const loadPrompts = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/prompts`);
      setPrompts(response.data.prompts);
      setPromptNames(response.data.prompt_names);
    } catch (error) {
      console.error('Error loading prompts:', error);
    }
  };

  const addPrompt = async (name, content) => {
    try {
      const promptId = Date.now().toString();
      await axios.post(`${API_BASE_URL}/api/prompts`, {
        id: promptId,
        name,
        content
      });
      await loadPrompts(); // Reload prompts
    } catch (error) {
      console.error('Error adding prompt:', error);
      throw error;
    }
  };

  const updatePrompt = async (id, name, content) => {
    try {
      await axios.put(`${API_BASE_URL}/api/prompts/${id}`, {
        id,
        name,
        content
      });
      await loadPrompts(); // Reload prompts
    } catch (error) {
      console.error('Error updating prompt:', error);
      throw error;
    }
  };

  const deletePrompt = async (id) => {
    try {
      await axios.delete(`${API_BASE_URL}/api/prompts/${id}`);
      await loadPrompts(); // Reload prompts
    } catch (error) {
      console.error('Error deleting prompt:', error);
      throw error;
    }
  };

  const runTest = async (selectedPrompts, mode, additionalParams = {}) => {
    if (!query.trim()) {
      alert('Please enter a query');
      return;
    }

    if (selectedPrompts.length === 0) {
      alert('Please select at least one prompt');
      return;
    }

    if (!apiConfig.api_url) {
      alert('Please configure API endpoint');
      return;
    }

    setLoading(true);
    
    try {
      const requestData = {
        query,
        api_config: apiConfig,
        prompts: selectedPrompts,
        mode: mode.toLowerCase().replace(' testing', '').replace(' ', '_'),
        ...additionalParams
      };

      let endpoint;
      switch (mode) {
        case 'Individual Testing':
          endpoint = '/api/test/individual';
          break;
        case 'Prompt Chaining':
          endpoint = '/api/test/chaining';
          break;
        case 'Prompt Combination':
          endpoint = '/api/test/combination';
          break;
        default:
          throw new Error('Unknown test mode');
      }

      const response = await axios.post(`${API_BASE_URL}${endpoint}`, requestData);
      
      // Update results based on mode
      if (mode === 'Individual Testing') {
        setTestResults(prev => [...prev, ...response.data.results]);
      } else if (mode === 'Prompt Chaining') {
        setChainResults(prev => [...prev, ...response.data.results]);
      } else if (mode === 'Prompt Combination') {
        setCombinationResults(prev => ({
          ...prev,
          [response.data.result.unique_id]: response.data.result
        }));
      }
      
    } catch (error) {
      console.error('Error running test:', error);
      alert(`Test failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const updateResult = async (resultId, updates) => {
    try {
      await axios.put(`${API_BASE_URL}/api/results/${resultId}`, updates);
      
      // Update local state
      setTestResults(prev => 
        prev.map(result => 
          result.unique_id === resultId 
            ? { ...result, ...updates, edited: updates.response ? true : result.edited }
            : result
        )
      );
      
      setChainResults(prev => 
        prev.map(result => 
          result.unique_id === resultId 
            ? { ...result, ...updates, edited: updates.response ? true : result.edited }
            : result
        )
      );
      
      setCombinationResults(prev => {
        if (prev[resultId]) {
          return {
            ...prev,
            [resultId]: { ...prev[resultId], ...updates, edited: updates.response ? true : prev[resultId].edited }
          };
        }
        return prev;
      });
      
    } catch (error) {
      console.error('Error updating result:', error);
      alert(`Update failed: ${error.response?.data?.detail || error.message}`);
    }
  };

  const generateSuggestions = async (responseText) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/suggest-prompt`, {
        response: responseText
      });
      return response.data.suggestions;
    } catch (error) {
      console.error('Error generating suggestions:', error);
      return ['Error generating suggestions'];
    }
  };

  const exportData = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/export`);
      const data = response.data.export_data;
      
      // Convert to CSV
      if (data.length === 0) {
        alert('No data to export');
        return;
      }
      
      const headers = Object.keys(data[0]);
      const csvContent = [
        headers.join(','),
        ...data.map(row => 
          headers.map(header => {
            const value = row[header];
            // Escape commas and quotes in CSV
            if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
              return `"${value.replace(/"/g, '""')}"`;
            }
            return value || '';
          }).join(',')
        )
      ].join('\n');
      
      // Download file
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const link = document.createElement('a');
      const url = URL.createObjectURL(blob);
      link.setAttribute('href', url);
      link.setAttribute('download', `baingan_export_${new Date().toISOString().split('T')[0]}.csv`);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
    } catch (error) {
      console.error('Error exporting data:', error);
      alert(`Export failed: ${error.response?.data?.detail || error.message}`);
    }
  };

  const renderMainContent = () => {
    const commonProps = {
      prompts,
      query,
      apiConfig,
      loading,
      onRunTest: runTest,
      onUpdateResult: updateResult,
      onGenerateSuggestions: generateSuggestions
    };

    switch (testMode) {
      case 'Individual Testing':
        return (
          <IndividualMode
            {...commonProps}
            testResults={testResults}
          />
        );
      case 'Prompt Chaining':
        return (
          <ChainingMode
            {...commonProps}
            chainResults={chainResults}
          />
        );
      case 'Prompt Combination':
        return (
          <CombinationMode
            {...commonProps}
            combinationResults={combinationResults}
          />
        );
      default:
        return <div>Unknown test mode</div>;
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🔮 BainGan 🧆</h1>
        <p>Test <strong>system prompts</strong>, create <strong>prompt chains</strong>, and <strong>combine prompts</strong> with AI assistance</p>
      </header>

      <div className="app-body">
        <div className="sidebar">
          <Sidebar
            apiConfig={apiConfig}
            onApiConfigChange={setApiConfig}
            query={query}
            onQueryChange={setQuery}
            testMode={testMode}
            onTestModeChange={setTestMode}
          />
        </div>

        <div className="main-content">
          <div className="content-columns">
            <div className="left-column">
              <PromptManager
                prompts={prompts}
                promptNames={promptNames}
                onAddPrompt={addPrompt}
                onUpdatePrompt={updatePrompt}
                onDeletePrompt={deletePrompt}
              />
            </div>

            <div className="right-column">
              {renderMainContent()}
            </div>
          </div>
          
          <div className="export-section">
            <ExportSection
              onExport={exportData}
              hasData={testResults.length > 0 || chainResults.length > 0 || Object.keys(combinationResults).length > 0}
            />
          </div>
        </div>
      </div>

      <footer className="app-footer">
        <div className="features-info">
          <h3>💡 Enhanced Features:</h3>
          <ul>
            <li><strong>✏️ Prompt Management:</strong> Add, edit, and name prompts in all test modes</li>
            <li><strong>🧪 Individual Testing:</strong> Test multiple system prompts independently with editable responses</li>
            <li><strong>🔗 Prompt Chaining:</strong> Chain prompts where each step uses the previous output</li>
            <li><strong>🤝 Prompt Combination:</strong> Use AI to intelligently combine multiple prompts with auto-adjusting weights</li>
            <li><strong>🎚️ Slider Strategy:</strong> Custom influence weights for prompt combination, auto-adjusted to sum to 100%</li>
            <li><strong>🌡️ Temperature Control:</strong> 0-100% slider to control AI creativity for prompt combination</li>
            <li><strong>🔮 Smart Suggestions:</strong> Generate prompt suggestions with options to save, save and run, or edit</li>
            <li><strong>⭐ Response Rating:</strong> Rate all responses (0-10, stored as percentage in export)</li>
            <li><strong>📊 Comprehensive Export:</strong> All results including individual, chain, and combination data with ratings and remarks</li>
            <li><strong>💾 Response Editing:</strong> Edit and save responses, with reverse prompt engineering</li>
          </ul>
        </div>
      </footer>
    </div>
  );
};

export default App;