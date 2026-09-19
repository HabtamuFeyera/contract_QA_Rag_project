import React from 'react';

const CitationDrawer = ({ citation, onClose }) => {
  return (
    <aside className="lexi-inspector-panel">
      <div className="inspector-header">
        <div className="inspector-title">
          <span>🔍</span>
          <span>Clause Inspector</span>
        </div>
        <button className="close-drawer-btn" onClick={onClose} title="Close Inspector">
          ✕
        </button>
      </div>

      {citation ? (
        <div className="citation-inspect-card">
          <div className="inspect-source-title">
            📄 {citation.source}
          </div>

          <div className="inspect-meta-tags">
            <span className="inspect-meta-tag">Page {citation.page}</span>
            <span className="inspect-meta-tag">Chunk #{citation.chunk_id ?? citation.id}</span>
            {citation.bm25_score && (
              <span className="inspect-meta-tag">BM25: {citation.bm25_score.toFixed(2)}</span>
            )}
          </div>

          <div className="inspect-verified-stamp">
            <span>✓</span> Ground-Truth Verified Excerpt
          </div>

          <div className="inspect-quote-box">
            "{citation.snippet}"
          </div>

          <div style={{ fontSize: '11px', color: '#94a3b8', lineHeight: '1.4' }}>
            <em>Note: This excerpt was retrieved via Reciprocal Rank Fusion (RRF) combining dense semantic search and sparse BM25 keyword matching.</em>
          </div>
        </div>
      ) : (
        <div className="inspector-empty-state">
          <div className="empty-glass-icon">⚖️</div>
          <p style={{ fontWeight: 600, color: '#94a3b8', marginBottom: '6px' }}>
            No Clause Selected
          </p>
          <p>
            Click any <span style={{ color: '#38bdf8' }}>[Page X]</span> citation chip inside an AI response to inspect the verbatim legal text and grounding metrics.
          </p>
        </div>
      )}
    </aside>
  );
};

export default CitationDrawer;
