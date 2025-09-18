import React, { useState, useEffect } from 'react';
import axios from 'axios';

const IndividualMode = ({ 
  prompts, 
  query, 
  apiConfig, 
  loading, 
  onAddPrompt
}) => {
  const [selectedPrompts, setSelectedPrompts] = useState([]);
  const [testResults, setTestResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [responseRatings, setResponseRatings] = useState({});
  const [suggestions, setSuggestions] = useState({});
  const [loadingSuggestions, setLoadingSuggestions] = useState({});

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const handlePromptSelection = (promptContent, isSelected) => {
    if (isSelected) {
      setSelectedPrompts(prev => [...prev, promptContent]);
    } else {
      setSelectedPrompts(prev => prev.filter(p => p !== promptContent));
    }
  };

  const handleSelectAll = () => {
    if (selectedPrompts.length === prompts.length) {
      setSelectedPrompts([]);
    } else {
      setSelectedPrompts(prompts.map(p => p.content));
    }
  };

  const testAllPrompts = async () => {
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

    setIsLoading(true);
    
    try {
      const response = await axios.post(`${API_BASE_URL}/api/individual/test-all`, {
        query,
        api_config: apiConfig,
        selected_prompts: selectedPrompts
      });

      const newResults = response.data.results;
      setTestResults(prev => [...prev, ...newResults]);
      
      // Initialize ratings for new results
      const newRatings = {};
      newResults.forEach(result => {
        newRatings[result.unique_id] = result.rating || 0;
      });
      setResponseRatings(prev => ({ ...prev, ...newRatings }));

      alert(`Tested ${response.data.total_tested} prompts successfully!`);
      
    } catch (error) {
      console.error('Error testing prompts:', error);
      alert(`Test failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const updateResult = async (uniqueId, updates) => {
    try {
      await axios.put(`${API_BASE_URL}/api/individual/update-result`, {
        unique_id: uniqueId,
        ...updates
      });

      // Update local state
      setTestResults(prev => 
        prev.map(result => 
          result.unique_id === uniqueId 
            ? { ...result, ...updates, edited: updates.response ? true : result.edited }
            : result
        )
      );

      if (updates.rating !== undefined) {
        setResponseRatings(prev => ({ ...prev, [uniqueId]: updates.rating }));
      }

    } catch (error) {
      console.error('Error updating result:', error);
      alert(`Update failed: ${error.response?.data?.detail || error.message}`);
    }
  };

  const generateSuggestion = async (uniqueId, responseText, originalQuery) => {
    setLoadingSuggestions(prev => ({ ...prev, [uniqueId]: true }));
    
    try {
      const response = await axios.post(`${API_BASE_URL}/api/individual/suggest-prompt`, {
        response_text: responseText,
        original_query: originalQuery
      });

      setSuggestions(prev => ({
        ...prev,
        [uniqueId]: {
          text: response.data.suggestion,
          name: `Suggested Prompt ${Object.keys(prev).length + 1}`
        }
      }));

    } catch (error) {
      console.error('Error generating suggestion:', error);
      alert(`Suggestion failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoadingSuggestions(prev => ({ ...prev, [uniqueId]: false }));
    }
  };

  const saveSuggestedPrompt = async (uniqueId, name, content, runImmediately = false) => {
    try {
      const requestData = {
        name,
        content,
        run_immediately: runImmediately
      };

      if (runImmediately) {
        requestData.query = query;
        requestData.api_config = apiConfig;
      }

      const response = await axios.post(`${API_BASE_URL}/api/individual/save-suggested-prompt`, requestData);

      if (runImmediately && response.data.executed) {
        // Add new result to the list
        const newResult = {
          unique_id: response.data.unique_id,
          test_type: 'Individual',
          prompt_name: name,
          system_prompt: content,
          query: query,
          response: response.data.response,
          status: response.data.status,
          status_code: response.data.status || 'N/A',
          timestamp: new Date().toISOString(),
          rating: 0,
          remark: 'Saved and ran',
          edited: false
        };

        setTestResults(prev => [...prev, newResult]);
        setResponseRatings(prev => ({ ...prev, [newResult.unique_id]: 0 }));
      }

      // Add to prompts list
      await onAddPrompt(name, content);

      // Clear suggestion
      setSuggestions(prev => {
        const newSuggestions = { ...prev };
        delete newSuggestions[uniqueId];
        return newSuggestions;
      });

      alert(response.data.message);

    } catch (error) {
      console.error('Error saving suggested prompt:', error);
      alert(`Save failed: ${error.response?.data?.detail || error.message}`);
    }
  };

  const allSelected = selectedPrompts.length === prompts.length && prompts.length > 0;
  const canRun = selectedPrompts.length > 0 && query.trim() && apiConfig.api_url;

  // Filter results for individual tests
  const individualResults = testResults.filter(result => result.test_type === 'Individual');
  const successCount = individualResults.filter(r => r.status === 'Success').length;

  return (
    <div className="individual-mode">
      <div className="mode-header">
        <h2>🧪 Individual Testing</h2>
        <p>Test multiple system prompts independently with the same query</p>
      </div>

      {/* Prompt Selection */}
      <div className="prompt-selection">
        <div className="selection-header">
          <h3>Select Prompts to Test</h3>
          <div className="selection-controls">
            <button 
              className="select-all-btn"
              onClick={handleSelectAll}
              disabled={prompts.length === 0}
            >
              {allSelected ? '☑️ Deselect All' : '☐ Select All'} ({selectedPrompts.length}/{prompts.length})
            </button>
            <button
              className="run-test-btn primary"
              onClick={testAllPrompts}
              disabled={isLoading || !canRun}
            >
              {isLoading ? '🔄 Testing...' : `🚀 Test ${selectedPrompts.length} Prompts`}
            </button>
          </div>
        </div>

        {prompts.length === 0 ? (
          <div className="empty-state">
            <p>No prompts available. Add some prompts first!</p>
          </div>
        ) : (
          <div className="prompts-grid">
            {prompts.map(prompt => {
              const isSelected = selectedPrompts.includes(prompt.content);
              return (
                <div 
                  key={prompt.id} 
                  className={`prompt-card ${isSelected ? 'selected' : ''}`}
                  onClick={() => handlePromptSelection(prompt.content, !isSelected)}
                >
                  <div className="prompt-card-header">
                    <span className="checkbox">
                      {isSelected ? '☑️' : '☐'}
                    </span>
                    <h4>{prompt.name}</h4>
                  </div>
                  <p className="prompt-preview">
                    {prompt.content.length > 100 
                      ? prompt.content.substring(0, 100) + '...'
                      : prompt.content
                    }
                  </p>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Loading Indicator */}
      {isLoading && (
        <div className="loading-indicator">
          <div className="loading-spinner"></div>
          <p>Testing {selectedPrompts.length} prompts...</p>
        </div>
      )}

      {/* Results Section */}
      <div className="test-results">
        <div className="results-header">
          <h3>Saved Results ({individualResults.length})</h3>
          {individualResults.length > 0 && (
            <div className="results-stats">
              <div className="stat success">✅ Success: {successCount}/{individualResults.length}</div>
              <div className="stat error">❌ Errors: {individualResults.length - successCount}</div>
              <div className="stat rated">⭐ Rated: {Object.keys(responseRatings).filter(id => responseRatings[id] > 0).length}</div>
            </div>
          )}
        </div>

        {individualResults.length === 0 ? (
          <div className="empty-results">
            <p>No results to display yet. Run some tests first!</p>
          </div>
        ) : (
          <div className="results-container">
            {individualResults.map((result, index) => (
              <ResultCard
                key={result.unique_id}
                result={result}
                index={index}
                rating={responseRatings[result.unique_id] || 0}
                suggestion={suggestions[result.unique_id]}
                loadingSuggestion={loadingSuggestions[result.unique_id] || false}
                onUpdateResult={updateResult}
                onGenerateSuggestion={generateSuggestion}
                onSaveSuggestion={saveSuggestedPrompt}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

// ResultCard component
const ResultCard = ({ 
  result, 
  index, 
  rating, 
  suggestion, 
  loadingSuggestion,
  onUpdateResult,
  onGenerateSuggestion,
  onSaveSuggestion
}) => {
  const [editedResponse, setEditedResponse] = useState(result.response);
  const [currentRating, setCurrentRating] = useState(rating);
  const [remark, setRemark] = useState(result.remark || '');
  const [suggestedName, setSuggestedName] = useState('');
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    if (suggestion) {
      setSuggestedName(suggestion.name);
    }
  }, [suggestion]);

  const handleSaveEdit = () => {
    onUpdateResult(result.unique_id, {
      response: editedResponse,
      rating: currentRating,
      remark: remark
    });
  };

  const handleRatingChange = (newRating) => {
    setCurrentRating(newRating);
    onUpdateResult(result.unique_id, { rating: newRating });
  };

  const statusIcon = result.status === 'Success' ? '🟢' : '🔴';
  const hasChanges = editedResponse !== result.response || currentRating !== rating || remark !== (result.remark || '');

  return (
    <div className="result-card">
      <div className="result-header" onClick={() => setIsExpanded(!isExpanded)}>
        <div className="result-title">
          <span className="status-icon">{statusIcon}</span>
          <h4>{result.prompt_name} — {result.status}</h4>
        </div>
        <span className="expand-icon">{isExpanded ? '▼' : '▶'}</span>
      </div>

      {isExpanded && (
        <div className="result-content">
          <div className="content-section">
            <label>System Prompt:</label>
            <pre className="code-block">{result.system_prompt}</pre>
          </div>

          <div className="content-section">
            <label>Query:</label>
            <pre className="code-block">{result.query}</pre>
          </div>

          <div className="content-section">
            <label>Response (editable):</label>
            <textarea
              value={editedResponse}
              onChange={(e) => setEditedResponse(e.target.value)}
              rows="6"
              className="response-textarea"
            />
          </div>

          <div className="content-section">
            <label>Rate this response (0-10):</label>
            <div className="rating-controls">
              <input
                type="range"
                min="0"
                max="10"
                value={currentRating}
                onChange={(e) => handleRatingChange(parseInt(e.target.value))}
                className="rating-slider"
              />
              <span className="rating-value">{currentRating}/10</span>
            </div>
          </div>

          <div className="content-section">
            <label>Remark:</label>
            <input
              type="text"
              value={remark}
              onChange={(e) => setRemark(e.target.value)}
              placeholder="Add a remark..."
              className="remark-input"
            />
          </div>

          {hasChanges && (
            <button className="save-btn" onClick={handleSaveEdit}>
              💾 Save Changes
            </button>
          )}

          <div className="actions-section">
            <button
              className="suggest-btn"
              onClick={() => onGenerateSuggestion(result.unique_id, editedResponse, result.query)}
              disabled={loadingSuggestion}
            >
              {loadingSuggestion ? '🔄 Generating...' : '🔮 Suggest Prompt'}
            </button>
          </div>

          {suggestion && (
            <div className="suggestion-section">
              <h5>Suggested System Prompt:</h5>
              <textarea
                value={suggestion.text}
                readOnly
                rows="4"
                className="suggestion-textarea"
              />
              
              <div className="suggestion-controls">
                <input
                  type="text"
                  value={suggestedName}
                  onChange={(e) => setSuggestedName(e.target.value)}
                  placeholder="Name this suggested prompt..."
                  className="suggestion-name-input"
                />
                
                <div className="suggestion-buttons">
                  <button
                    className="save-only-btn"
                    onClick={() => onSaveSuggestion(result.unique_id, suggestedName, suggestion.text, false)}
                  >
                    💾 Save Only
                  </button>
                  <button
                    className="save-run-btn"
                    onClick={() => onSaveSuggestion(result.unique_id, suggestedName, suggestion.text, true)}
                  >
                    🏃 Save & Run
                  </button>
                </div>
              </div>
            </div>
          )}

          <div className="result-details">
            <small>
              Status Code: {result.status_code} | 
              Time: {new Date(result.timestamp).toLocaleString()} | 
              Rating: {currentRating}/10
              {result.edited && ' | ✏️ Edited'}
            </small>
          </div>
        </div>
      )}
    </div>
  );
};

export default IndividualMode;