import React, { useState, useEffect, useRef } from 'react';

const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

const SAMPLE_QUESTIONS = [
  "Who are the parties to the Agreement and what are their defined names?",
  "What are the payments to the Advisor under the Agreement?",
  "Is there a non-compete obligation to the Advisor?",
  "Whose consent is required for the assignment of the Agreement by the Buyer?",
  "How much is the escrow amount and what is its purpose?"
];

const Chatbot = () => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([
    {
      sender: 'bot',
      text: "👋 Welcome to **LexiRAG**! I am your Autonomous Legal Contract Advisor. Ask any question about your agreements, or upload a new PDF contract below.",
      citations: []
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [activeCitation, setActiveCitation] = useState(null);
  const [backendHealthy, setBackendHealthy] = useState(true);

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // Check health on load
  useEffect(() => {
    fetch(`${API_BASE_URL}/health`)
      .then((res) => res.json())
      .then(() => setBackendHealthy(true))
      .catch(() => setBackendHealthy(false));
  }, []);

  const handleSendMessage = async (queryText = null) => {
    const questionToSend = queryText || input;
    if (!questionToSend.trim() || isLoading) return;

    // Add user message
    const newMessages = [...messages, { sender: 'user', text: questionToSend }];
    setMessages(newMessages);
    setInput('');
    setIsLoading(true);
    setActiveCitation(null);

    try {
      const response = await fetch(`${API_BASE_URL}/query/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: questionToSend, top_k: 4 })
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        {
          sender: 'bot',
          text: data.answer || "No response generated.",
          citations: data.citations || []
        }
      ]);
    } catch (error) {
      console.error("Query error:", error);
      setMessages((prev) => [
        ...prev,
        {
          sender: 'bot',
          text: `⚠️ **Connection Error**: Unable to reach LexiRAG API at \`${API_BASE_URL}\`. Please ensure the backend is running.`,
          citations: []
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      alert("Please upload a valid PDF contract.");
      return;
    }

    setIsUploading(true);
    setUploadStatus(`Ingesting ${file.name}...`);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE_URL}/upload/`, {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed");
      }

      const data = await res.json();
      setUploadStatus(`✅ ${file.name} indexed (${data.chunks_indexed} chunks)`);
      setMessages((prev) => [
        ...prev,
        {
          sender: 'bot',
          text: `📄 **Contract Ingested**: Successfully analyzed and indexed \`${file.name}\` (${data.pages_extracted} pages, ${data.chunks_indexed} clauses). You can now ask questions about this document!`,
          citations: []
        }
      ]);
    } catch (err) {
      console.error("Upload error:", err);
      setUploadStatus(`❌ Ingestion failed: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const clearChat = () => {
    setMessages([
      {
        sender: 'bot',
        text: "Conversation cleared. Ready for your next contract inquiry.",
        citations: []
      }
    ]);
    setActiveCitation(null);
  };

  return (
    <div className="App">
      {/* Navigation Header */}
      <header className="lexi-header">
        <div className="lexi-logo-area">
          <div className="lexi-badge-icon">⚖️</div>
          <div>
            <h1 className="lexi-title">LexiRAG</h1>
            <p className="lexi-subtitle">Autonomous Legal Intelligence & Clause Verification Engine</p>
          </div>
        </div>

        <div className="lexi-status-bar">
          <div className="status-indicator">
            <span className="status-dot" style={{ backgroundColor: backendHealthy ? '#10b981' : '#ef4444' }}></span>
            <span>{backendHealthy ? 'ChromaDB & LLM Online' : 'Backend Disconnected'}</span>
          </div>
          <button className="clear-btn" onClick={clearChat} title="Clear Chat History">
            🧹 Reset
          </button>
        </div>
      </header>

      {/* Main Split Workspace */}
      <div className="lexi-workspace">
        {/* Left Sidebar: Document Management & Quick Queries */}
        <aside className="lexi-sidebar">
          {/* Upload Card */}
          <div className="sidebar-section">
            <div className="section-heading">
              <span>📄</span> Ingest Contract PDF
            </div>
            <label className="dropzone">
              <div className="upload-icon">📥</div>
              <div className="upload-title">
                {isUploading ? "Processing Document..." : "Click or Drag PDF"}
              </div>
              <div className="upload-desc">
                Supports legal agreements, bylaws & amendments
              </div>
              <input 
                type="file" 
                accept="application/pdf" 
                onChange={handleFileUpload} 
                disabled={isUploading} 
              />
            </label>
            {uploadStatus && (
              <p style={{ fontSize: '11px', marginTop: '8px', color: '#38bdf8' }}>
                {uploadStatus}
              </p>
            )}
          </div>

          {/* Quick Legal Inquiries */}
          <div className="sidebar-section" style={{ flex: 1 }}>
            <div className="section-heading">
              <span>💡</span> Sample Contract Queries
            </div>
            <div className="prompt-chips">
              {SAMPLE_QUESTIONS.map((q, idx) => (
                <button
                  key={idx}
                  className="prompt-chip"
                  onClick={() => handleSendMessage(q)}
                  disabled={isLoading}
                >
                  <span>⚖️</span> {q}
                </button>
              ))}
            </div>
          </div>
        </aside>

        {/* Right Panel: Chat Workspace */}
        <main className="chat-container">
          <div className="chat-messages">
            {messages.map((msg, index) => (
              <div key={index} className={`message-row ${msg.sender}`}>
                <div className={`avatar ${msg.sender}`}>
                  {msg.sender === 'user' ? '👤' : '🤖'}
                </div>
                <div className="bubble">
                  <div style={{ whiteSpace: 'pre-line' }}>{msg.text}</div>

                  {/* Ground Truth / Clause Citations */}
                  {msg.citations && msg.citations.length > 0 && (
                    <div className="citations-box">
                      <div className="citations-label">
                        <span>🔍</span> Verified Clause Citations ({msg.citations.length}):
                      </div>
                      <div className="citation-chips">
                        {msg.citations.map((cite) => (
                          <span
                            key={cite.id}
                            className="citation-pill"
                            onClick={() =>
                              setActiveCitation(
                                activeCitation?.id === cite.id ? null : cite
                              )
                            }
                            title="Click to view extracted clause text"
                          >
                            [Page {cite.page}: {cite.source}]
                          </span>
                        ))}
                      </div>

                      {/* Expanded Citation Preview */}
                      {activeCitation && (
                        <div className="citation-detail">
                          <strong>Source:</strong> {activeCitation.source} (Page {activeCitation.page})<br/>
                          <strong>Excerpt:</strong> "{activeCitation.snippet}"
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="message-row bot">
                <div className="avatar bot">🤖</div>
                <div className="bubble" style={{ color: '#38bdf8' }}>
                  <span>⏳ Cross-examining contract clauses & generating verified citations...</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Box */}
          <div className="chat-input-bar">
            <input
              type="text"
              placeholder="Ask a legal or contractual question (e.g. 'What are the liability limits?')..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
              disabled={isLoading}
            />
            <button
              className="send-btn"
              onClick={() => handleSendMessage()}
              disabled={isLoading || !input.trim()}
            >
              <span>Send Query</span> ➔
            </button>
          </div>
        </main>
      </div>
    </div>
  );
};

export default Chatbot;
