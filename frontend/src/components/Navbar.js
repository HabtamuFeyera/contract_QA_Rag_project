import React from 'react';

const Navbar = ({ backendHealthy, onResetChat, activeDrawer, onToggleDrawer }) => {
  return (
    <header className="lexi-navbar">
      <div className="nav-brand-section">
        <div className="brand-icon-shield">⚖️</div>
        <div className="brand-text-block">
          <div className="brand-title-row">
            <span className="brand-name">LexiRAG</span>
            <span className="enterprise-tag">Enterprise AI</span>
          </div>
          <span className="brand-tagline">Autonomous Legal Intelligence & Clause Verification Engine</span>
        </div>
      </div>

      <div className="nav-controls-section">
        <div className="health-pill" title="Real-time backend & ChromaDB status">
          <span 
            className="pulse-dot" 
            style={{ backgroundColor: backendHealthy ? '#10b981' : '#ef4444' }}
          />
          <span>{backendHealthy ? 'ChromaDB & LLM Online' : 'Backend Disconnected'}</span>
        </div>

        <div className="rag-mode-badge" title="Retrieval configuration">
          <span>Mode: Hybrid (Dense + BM25 RRF)</span>
        </div>

        <button 
          className="action-btn-secondary" 
          onClick={onToggleDrawer}
          title="Toggle Clause Citation Drawer"
        >
          {activeDrawer ? 'Hide Inspector ⇥' : '🔍 Clause Inspector'}
        </button>

        <button 
          className="action-btn-secondary" 
          onClick={onResetChat}
          title="Clear conversational memory"
        >
          🧹 Reset
        </button>
      </div>
    </header>
  );
};

export default Navbar;
