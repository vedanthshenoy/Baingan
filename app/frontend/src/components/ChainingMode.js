import React, { useState, useEffect } from 'react';
import axios from 'axios';

const ChainingMode = ({ 
  prompts, 
  query, 
  apiConfig, 
  onAddPrompt
}) => {
  const [chainResults, setChainResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [responseRatings, setResponseRatings] = useState({});
  const [suggestions, setSuggestions] = useState({});
  const [loadingSuggestions, setLoadingSuggestions] = useState({});
  const [editingSuggestions, setEditingSuggestions] = useState({});

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const testChainedPrompts = async () => {
    if (!query.trim()) {
      alert('Please enter a query');
      return;
    }

    if (prompts.length === 0) {
      alert('Please add at least one prompt');
      return;
    }

    if (!apiConfig.api_url) {
      alert('Please configure API endpoint');
      return;
    }

    setIsLoading(true);
    
    try {
      const response = await axios.post(`${API_BASE_URL}/api/chaining/test-chained`, {
        query,
        api_config: apiConfig,
        prompts: prompts.map(p => ({ name: p.name, content: p.content }))
      });

      const newResults = response.data.results;
      setChainResults(prev => [...prev, ...newResults]);
      
      // Initialize ratings for new results
      const newRatings = {};
      newResults.forEach(result => {
        newRatings[result.unique_id] = result.rating || 0;
      });
      setResponseRatings(prev => ({ ...prev, ...newRatings }));

      alert(`Completed chain with ${response.data.total_steps} steps!`);
      
    } catch (error) {
      console.error('Error testing chained prompts:', error);
      alert(`Chain test failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const updateResult = async (uniqueId, updates) => {
    try {
      await axios.put(`${API_BASE_URL}/api/chaining/update-result`, {
        unique_id: uniqueId,
        ...updates
      });

      // Update local state
      setChainResults(prev => 
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
      const response = await axios.post(`${API_BASE_URL}/api/chaining/suggest-prompt`, {
        response_text: responseText,
        original_query: originalQuery
      });

      const result = chainResults.find(r => r.unique_id === uniqueId);
      setSuggestions(prev => ({
        ...prev,
        [uniqueId]: {
          text: response.data.suggestion,
          name: `Suggested Prompt Step ${result?.step || 'Unknown'}`
        }
      }));

    } catch (error) {
      console.error('Error generating suggestion:', error);
      alert(`Suggestion failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoadingSuggestions(prev => ({ ...prev, [uniqueId]: false }));
    }
  };

  const generateReversePrompt = async (uniqueId, editedResponse) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/chaining/reverse-prompt`, {
        unique_id: uniqueId,
        response: editedResponse
      });

      // Update the result with the suggested system prompt
      await updateResult(uniqueId, {
        system_prompt: response.data.suggested_prompt
      });

      alert('System prompt updated based on edited response!');

    } catch (error) {
      console.error('Error generating reverse prompt:', error);
      alert(`Reverse prompt failed: ${error.response?.data?.detail || error.message}`);
    }
  };

  const saveSuggestedPrompt = async (uniqueId, name, content, action) => {
    try {
      const result = chainResults.find(r => r.unique_id === uniqueId);
      
      const requestData = {
        name,
        content,
        action,
        step: result?.step,
        input_query: result?.input_query
      };

      if (action === 'save_and_run') {
        requestData.query = result?.query;
        requestData.api_config = apiConfig;
      }

      const response = await axios.post(`${API_BASE_URL}/api/chaining/save-suggested-prompt`, requestData);

      if (action === 'save_and_run' && response.data.executed) {
        // Add new result to the list
        const newResult = {
          unique_id: response.data.unique_id,
          test_type: 'Chaining',
          prompt_name: name,
          system_prompt: content,
          query: result?.query,
          response: response.data.response,
          status: response.data.status,
          status_code: response.data.status_code || 'N/A',
          timestamp: new Date().toISOString(),
          rating: 0,
          remark: `Saved and ran for step ${result?.step}`,
          edited: false,
          step: result?.step,
          input_query: result?.input_query
        };

        setChainResults(prev => [...prev, newResult]);
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

      // Clear editing state if it was being edited
      setEditingSuggestions(prev => {
        const newEditing = { ...prev };
        delete newEditing[uniqueId];
        return newEditing;
      });

      alert(response.data.message);

    } catch (error) {
      console.error('Error saving suggested prompt:', error);
      alert(`Save failed: ${error.response?.data?.detail || error.message}`);
    }
  };

  const canRun = prompts.length > 0 && query.trim() && apiConfig.api_url;
  const sortedResults = chainResults.sort((a, b) => (a.step || 0) - (b.step || 0));
  const successCount = chainResults.filter(r => r.status === 'Success').length;

  return (
    <div className="chaining-mode">
      <div className="mode-header">
        <h2>🔗 Prompt Chaining Testing</h2>
        <p>Chain prompts where each step uses the previous output as input</p>
      </div>

      {/* Test Button */}
      <div className="test-section">
        <div className="test-info">
          <h3>Chain Configuration</h3>
          <p>Ready to test {prompts.length} prompts in sequence</p>
          <div className="chain-preview">
            {prompts.map((prompt, index) => (
              <div key={prompt.id} className="chain-step-preview">
                <div className="step-number">{index + 1}</div>
                <div className="step-info">
                  <strong>{prompt.name}</strong>
                  <small>{prompt.content.substring(0, 60)}...</small>
                </div>
                {index < prompts.length - 1 && <div className="arrow">→</div>}
              </div>
            ))}
          </div>
        </div>

        <button
          className="run-chain-btn primary"
          onClick={testChainedPrompts}
          disabled={isLoading || !canRun}
        >
          {isLoading ? '🔄 Running Chain...' : `🚀 Test Chained Prompts (${prompts.length} steps)`}
        </button>
      </div>

      {/* Loading Indicator */}
      {isLoading && (
        <div className="loading-indicator">
          <div className="loading-spinner"></div>
          <p>Running chained prompts...</p>
        </div>
      )}

      {/* Results Section */}
      <div className="chain-results">
        <div className="results-header">
          <h3>Saved Chained Results ({chainResults.length})</h3>
          {chainResults.length > 0 && (
            <div className="results-stats">
              <div className="stat success">✅ Success: {successCount}/{chainResults.length}</div>
              <div className="stat error">❌ Errors: {chainResults.length - successCount}</div>
              <div className="stat rated">⭐ Rated: {Object.keys(responseRatings).filter(id => responseRatings[id] > 0).length}</div>
            </div>
          )}
        </div>

        {chainResults.length === 0 ? (
          <div className="empty-results">
            <p>No chained results to display yet. Run some chained tests first!</p>
          </div>
        ) : (
          <div className="results-container">
            {sortedResults.map((result) => (
              <ChainResultCard
                key={result.unique_id}
                result={result}
                rating={responseRatings[result.unique_id] || 0}
                suggestion={suggestions[result.unique_id]}
                loadingSuggestion={loadingSuggestions[result.unique_id] || false}
                editingSuggestion={editingSuggestions[result.unique_id]}
                onUpdateResult={updateResult}
                onGenerateSuggestion={generateSuggestion}
                onGenerateReversePrompt={generateReversePrompt}
                onSaveSuggestion={saveSuggestedPrompt}
                onEditSuggestion={(uniqueId, editing) => 
                  setEditingSuggestions(prev => ({ ...prev, [uniqueId]: editing }))
                }
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

// Chain Result Card Component
const ChainResultCard = ({ 
  result, 
  rating, 
  suggestion, 
  loadingSuggestion,
  editingSuggestion,
  onUpdateResult,
  onGenerateSuggestion,
  onGenerateReversePrompt,
  onSaveSuggestion,
  onEditSuggestion
}) => {
  const [editedResponse, setEditedResponse] = useState(result.response);
  const [currentRating, setCurrentRating] = useState(rating);
  const [remark, setRemark] = useState(result.remark || '');
  const [suggestedName, setSuggestedName] = useState('');
  const [editedSuggestion, setEditedSuggestion] = useState('');
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    if (suggestion) {
      setSuggestedName(suggestion.name);
      setEditedSuggestion(suggestion.text);
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

  const handleReversePrompt = () => {
    onGenerateReversePrompt(result.unique_id, editedResponse);
  };

  const statusIcon = result.status === 'Success' ? '🟢' : '🔴';
  const hasChanges = editedResponse !== result.response || currentRating !== rating || remark !== (result.remark || '');

  return (
    <div className="chain-result-card">
      <div className="result-header" onClick={() => setIsExpanded(!isExpanded)}>
        <div className="result-title">
          <span className="status-icon">{statusIcon}</span>
          <span className="step-badge">Step {result.step}</span>
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
            <label>Input Query:</label>
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
            <div className="save-section">
              <div className="save-buttons">
                <button className="save-btn" onClick={handleSaveEdit}>
                  💾 Save Edited Response
                </button>
                <button className="reverse-btn" onClick={handleReversePrompt}>
                  🔄 Reverse Prompt
                </button>
              </div>
            </div>
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
              
              {editingSuggestion ? (
                <div className="edit-suggestion">
                  <textarea
                    value={editedSuggestion}
                    onChange={(e) => setEditedSuggestion(e.target.value)}
                    rows="4"
                    className="suggestion-textarea"
                  />
                  
                  <input
                    type="text"
                    value={suggestedName}
                    onChange={(e) => setSuggestedName(e.target.value)}
                    placeholder="Name this edited prompt..."
                    className="suggestion-name-input"
                  />
                  
                  <div className="suggestion-buttons">
                    <button
                      className="save-only-btn"
                      onClick={() => {
                        onSaveSuggestion(result.unique_id, suggestedName, editedSuggestion, 'save_only');
                        onEditSuggestion(result.unique_id, false);
                      }}
                    >
                      💾 Save Edited
                    </button>
                    <button
                      className="cancel-edit-btn"
                      onClick={() => onEditSuggestion(result.unique_id, false)}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <div className="view-suggestion">
                  <textarea
                    value={suggestion.text}
                    readOnly
                    rows="4"
                    className="suggestion-textarea readonly"
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
                        onClick={() => onSaveSuggestion(result.unique_id, suggestedName, suggestion.text, 'save_only')}
                      >
                        💾 Save Only
                      </button>
                      <button
                        className="save-run-btn"
                        onClick={() => onSaveSuggestion(result.unique_id, suggestedName, suggestion.text, 'save_and_run')}
                      >
                        🏃 Save & Run
                      </button>
                      <button
                        className="edit-btn"
                        onClick={() => onEditSuggestion(result.unique_id, true)}
                      >
                        ✏️ Edit
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          <div className="result-details">
            <small>
              Status Code: {result.status_code} | 
              Time: {new Date(result.timestamp).toLocaleString()} | 
              Step: {result.step} | 
              Rating: {currentRating}/10 ({currentRating * 10}%)
              {result.edited && ' | ✏️ Edited'}
            </small>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChainingMode;