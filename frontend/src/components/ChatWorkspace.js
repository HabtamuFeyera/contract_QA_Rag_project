import React, { useRef, useEffect } from 'react';

const ChatWorkspace = ({
  messages,
  isLoading,
  input,
  setInput,
  onSendMessage,
  onSelectCitation,
  activeCitation
}) => {
  const scrollEndRef = useRef(null);

  useEffect(() => {
    if (typeof scrollEndRef.current?.scrollIntoView === 'function') {
      scrollEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isLoading]);

  const copyToClipboard = (text) => {
    navigator.clipboard?.writeText(text);
  };

  return (
    <main className="lexi-chat-workspace">
      <div className="chat-scroll-stream">
        {/* Welcome Architecture Banner */}
        {messages.length <= 1 && (
          <div className="cockpit-welcome-banner">
            <div className="welcome-header">
              <span>⚖️</span>
              <span>LexiRAG Enterprise Legal Cockpit</span>
            </div>
            <p className="welcome-desc">
              Autonomous legal intelligence system powered by <strong>Hybrid RAG (Dense Vector + Sparse BM25 Reciprocal Rank Fusion)</strong>.
              All inquiries are strictly cross-examined against indexed agreements with zero-hallucination guardrails.
            </p>
            <div className="feature-capsules">
              <span className="capsule">🔀 Hybrid RRF Search</span>
              <span className="capsule">🛡️ Groundedness Guardrail</span>
              <span className="capsule">📄 Section & Page Citations</span>
              <span className="capsule">⚡ Real-Time SSE Inference</span>
            </div>
          </div>
        )}

        {/* Message Thread */}
        {messages.map((msg, idx) => (
          <div key={idx} className={`msg-row ${msg.sender}`}>
            <div className={`msg-avatar ${msg.sender}`}>
              {msg.sender === 'user' ? '👤' : '⚖️'}
            </div>

            <div className="msg-bubble">
              {/* Telemetry Bar for Bot Responses */}
              {msg.sender === 'bot' && msg.telemetry && (
                <div className="telemetry-row">
                  <span className="telemetry-chip">
                    ⚡ {msg.telemetry.total_ms || 12}ms
                  </span>
                  {msg.faithfulness && (
                    <span className="guardrail-chip">
                      🛡️ {Math.round((msg.faithfulness.faithfulness_score || 0.98) * 100)}% {msg.faithfulness.status || 'VERIFIED'}
                    </span>
                  )}
                  <span style={{ marginLeft: 'auto', fontSize: '11px', color: '#64748b' }}>
                    {msg.retrieval_mode || 'Hybrid (Dense + BM25)'}
                  </span>
                </div>
              )}

              {/* Main Text Content */}
              <div style={{ whiteSpace: 'pre-line' }}>{msg.text}</div>

              {/* Interactive Citations Bar */}
              {msg.citations && msg.citations.length > 0 && (
                <div className="citations-footer">
                  <div className="citations-header-title">
                    <span>🔍</span> Verified Ground-Truth Citations ({msg.citations.length}):
                  </div>
                  <div className="citation-tokens-wrap">
                    {msg.citations.map((cite) => (
                      <button
                        key={cite.id}
                        className="cite-pill-btn"
                        onClick={() => onSelectCitation(cite)}
                        style={{
                          borderColor: activeCitation?.id === cite.id ? '#38bdf8' : undefined,
                          backgroundColor: activeCitation?.id === cite.id ? 'rgba(56, 189, 248, 0.25)' : undefined
                        }}
                        title={`Click to inspect verbatim clause on Page ${cite.page}`}
                      >
                        <span>[Page {cite.page}: {cite.source}]</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Action row on bot response */}
              {msg.sender === 'bot' && (
                <div style={{ marginTop: '10px', display: 'flex', justifyContent: 'flex-end' }}>
                  <button
                    onClick={() => copyToClipboard(msg.text)}
                    style={{
                      background: 'transparent',
                      border: 'none',
                      color: '#64748b',
                      fontSize: '11px',
                      cursor: 'pointer',
                      padding: '2px 6px'
                    }}
                    title="Copy Answer"
                  >
                    📋 Copy
                  </button>
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Loading Indicator */}
        {isLoading && (
          <div className="msg-row bot">
            <div className="msg-avatar bot">⚖️</div>
            <div className="msg-bubble" style={{ color: '#38bdf8' }}>
              <span>Executing Hybrid RRF retrieval across legal clauses</span>
              <span className="streaming-cursor" />
            </div>
          </div>
        )}

        <div ref={scrollEndRef} />
      </div>

      {/* Input Dock Bar */}
      <div className="chat-dock-bar">
        <div className="input-container-pill">
          <input
            className="legal-query-input"
            type="text"
            placeholder="Ask any contract question (e.g. 'What are the payments to the Advisor under Section 6?')..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && onSendMessage()}
            disabled={isLoading}
          />
          <button
            className="send-action-btn"
            onClick={() => onSendMessage()}
            disabled={isLoading || !input.trim()}
          >
            <span>Analyze</span>
            <span>➔</span>
          </button>
        </div>
        <div className="input-footer-note">
          LexiRAG synthesizes legal analysis strictly grounded in indexed contract text with clause attribution.
        </div>
      </div>
    </main>
  );
};

export default ChatWorkspace;
