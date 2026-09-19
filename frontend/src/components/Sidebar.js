import React from 'react';

const PRELOADED_CONTRACTS = [
  { id: 'raptor', name: 'Raptor Stock Purchase Agreement', type: 'M&A Agreement', pages: '48 pages', clauses: '270 clauses' },
  { id: 'robinson', name: 'Robinson Advisory Agreement', type: 'Executive Retainer', pages: '8 pages', clauses: '16 clauses' },
  { id: 'raptor_qa', name: 'Raptor Q&A Guidelines', type: 'Disclosure Schedule', pages: '2 pages', clauses: '4 clauses' },
  { id: 'robinson_qa', name: 'Robinson Undertaking & Q&A', type: 'IP & Non-Compete', pages: '2 pages', clauses: '4 clauses' }
];

const LEGAL_QUERIES = [
  { category: '💰 Compensation & Escrow', question: 'What are the payments to the Advisor under Section 6?' },
  { category: '💰 Compensation & Escrow', question: 'How much is the escrow amount and what is its purpose?' },
  { category: '🔒 Restrictive Covenants', question: 'Is there a non-compete obligation to the Advisor?' },
  { category: '📜 Assignment & Parties', question: 'Who are the parties to the Agreement and what are their defined names?' },
  { category: '📜 Assignment & Parties', question: 'Whose consent is required for the assignment of the Agreement by the Buyer?' }
];

const Sidebar = ({
  activeContract,
  onSelectContract,
  onSelectQuery,
  onFileUpload,
  isUploading,
  uploadStatus
}) => {
  return (
    <aside className="lexi-sidebar-panel">
      {/* Contract Repository Explorer */}
      <div className="sidebar-card">
        <div className="sidebar-title">
          <span>📁 Legal Repositories</span>
          <span style={{ fontSize: '10px', color: '#94a3b8' }}>4 Indexed</span>
        </div>
        <div className="contract-list">
          {PRELOADED_CONTRACTS.map((contract) => (
            <div
              key={contract.id}
              className={`contract-item ${activeContract === contract.id ? 'active' : ''}`}
              onClick={() => onSelectContract(contract.id)}
            >
              <div className="contract-meta">
                <span className="contract-name" title={contract.name}>
                  {contract.name}
                </span>
                <span className="contract-subtext">
                  {contract.type} • {contract.clauses}
                </span>
              </div>
              <span className="contract-badge-status">●</span>
            </div>
          ))}
        </div>
      </div>

      {/* Contract Upload Dropzone */}
      <div className="sidebar-card">
        <div className="sidebar-title">
          <span>📄 Ingest New PDF</span>
        </div>
        <label className="upload-dropzone">
          <div className="drop-icon">📥</div>
          <div className="drop-text-main">
            {isUploading ? 'Parsing & Indexing...' : 'Click or Drag PDF'}
          </div>
          <div className="drop-text-sub">
            Extracts clauses & builds BM25 + Vector index
          </div>
          <input
            type="file"
            accept="application/pdf"
            onChange={onFileUpload}
            disabled={isUploading}
          />
        </label>
        {uploadStatus && (
          <p style={{ fontSize: '11px', marginTop: '8px', color: '#38bdf8', textAlign: 'center' }}>
            {uploadStatus}
          </p>
        )}
      </div>

      {/* Pre-Engineered Legal Inquiries */}
      <div className="sidebar-card" style={{ flex: 1 }}>
        <div className="sidebar-title">
          <span>⚖️ Suggested Inquiries</span>
        </div>
        <div className="query-pills-list">
          {LEGAL_QUERIES.map((item, idx) => (
            <button
              key={idx}
              className="query-pill"
              onClick={() => onSelectQuery(item.question)}
              title={item.category}
            >
              <span>▸</span>
              <span>{item.question}</span>
            </button>
          ))}
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
