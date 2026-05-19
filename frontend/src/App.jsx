import React, { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const SAMPLE_QUERIES = [
  "What are the RRC states in 5G NR?",
  "Explain handover procedure in NR",
  "What is carrier aggregation in NR?",
  "How does HARQ work in 5G?",
  "What is the role of gNB-DU and gNB-CU?",
  "Describe the RRC connection setup procedure"
]

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [specFilter, setSpecFilter] = useState('')
  const [releaseFilter, setReleaseFilter] = useState('')
  const messagesEndRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendQuery = async (query) => {
    if (!query.trim()) return
    const userMsg = { role: 'user', content: query, timestamp: new Date() }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)

    try {
      const res = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          spec_filter: specFilter || null,
          release_filter: releaseFilter || null
        })
      })
      const data = await res.json()
      const assistantMsg = {
        role: 'assistant',
        content: data.answer,
        citations: data.citations,
        confidence: data.confidence,
        latency_ms: data.latency_ms,
        chunks_retrieved: data.chunks_retrieved,
        timestamp: new Date()
      }
      setMessages(prev => [...prev, assistantMsg])
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `**Error:** ${err.message}\n\nMake sure the backend is running:\n\`\`\`bash\ncd backend && uvicorn api:app --reload --port 8000\n\`\`\``,
        timestamp: new Date()
      }])
    }
    setLoading(false)
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    sendQuery(input)
  }

  const clearChat = () => setMessages([])

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <span className="logo-icon">📡</span>
            <h1>3GPP RAG Expert</h1>
            <span className="badge">Powered by Amazon Bedrock</span>
          </div>
          <div className="header-right">
            <span className="status-badge">● 15K+ chunks indexed</span>
            <button className="clear-btn" onClick={clearChat}>Clear Chat</button>
          </div>
        </div>
      </header>

      <div className="main-container">
        <aside className="sidebar">
          <div className="sidebar-section">
            <h3>🔍 Filters</h3>
            <label>Spec Number</label>
            <input
              type="text"
              placeholder="e.g. 38331"
              value={specFilter}
              onChange={(e) => setSpecFilter(e.target.value)}
            />
            <label>Release</label>
            <input
              type="text"
              placeholder="e.g. Rel-18"
              value={releaseFilter}
              onChange={(e) => setReleaseFilter(e.target.value)}
            />
          </div>

          <div className="sidebar-section">
            <h3>⚡ Quick Queries</h3>
            {SAMPLE_QUERIES.map((q, i) => (
              <button key={i} className="sample-btn" onClick={() => sendQuery(q)}>
                <span className="sample-icon">→</span> {q}
              </button>
            ))}
          </div>

          <div className="sidebar-section">
            <h3>🏗️ Pipeline</h3>
            <div className="pipeline-steps">
              <div className="step">
                <span className="step-num">1</span>
                <span>Query Decomposition</span>
              </div>
              <div className="step">
                <span className="step-num">2</span>
                <span>Multi-Query Retrieval</span>
              </div>
              <div className="step">
                <span className="step-num">3</span>
                <span>Score & Rerank</span>
              </div>
              <div className="step">
                <span className="step-num">4</span>
                <span>Expert Generation</span>
              </div>
            </div>
          </div>

          <div className="sidebar-section advantage-box">
            <h3>✅ vs Generic AI</h3>
            <ul>
              <li>Exact clause citations</li>
              <li>Zero hallucination</li>
              <li>Latest Rel-18/19/20</li>
              <li>Verifiable sources</li>
            </ul>
          </div>
        </aside>

        <main className="chat-area">
          <div className="messages">
            {messages.length === 0 && (
              <div className="welcome">
                <div className="welcome-icon">📡</div>
                <h2>3GPP Specification Expert</h2>
                <p>Ask any question about 5G NR, LTE, or 3GPP standards. Every answer is grounded in official specification text with exact clause citations.</p>
                <div className="welcome-features">
                  <div className="feature">
                    <span>🎯</span>
                    <strong>More Accurate</strong>
                    <p>Than ChatGPT/Gemini — grounded in exact spec text</p>
                  </div>
                  <div className="feature">
                    <span>📋</span>
                    <strong>Structured</strong>
                    <p>Tables, diagrams, protocol flows</p>
                  </div>
                  <div className="feature">
                    <span>🔗</span>
                    <strong>Traceable</strong>
                    <p>Every claim linked to TS clause</p>
                  </div>
                </div>
              </div>
            )}

            {messages.map((msg, i) => (
              <div key={i} className={`message ${msg.role}`}>
                <div className="message-avatar">
                  {msg.role === 'user' ? '👤' : '🤖'}
                </div>
                <div className="message-content">
                  <div className="message-bubble">
                    {msg.role === 'assistant' ? (
                      <div className="markdown-body">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {msg.content}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      <p>{msg.content}</p>
                    )}
                  </div>

                  {msg.role === 'assistant' && msg.citations && msg.citations.length > 0 && (
                    <div className="citations-panel">
                      <div className="citations-header">
                        <span>📚 Sources ({msg.citations.length})</span>
                      </div>
                      <div className="citations-list">
                        {msg.citations.map((c, j) => (
                          <div key={j} className="citation-card">
                            <span className="citation-spec">TS {c.spec}</span>
                            <span className="citation-section">§{c.section}</span>
                            <span className="citation-release">{c.release}</span>
                            <span className="citation-score">{c.score}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {msg.role === 'assistant' && msg.confidence !== undefined && (
                    <div className="meta-bar">
                      <div className={`confidence-badge ${msg.confidence >= 0.7 ? 'high' : msg.confidence >= 0.4 ? 'medium' : 'low'}`}>
                        <span className="conf-dot"></span>
                        Confidence: {Math.round(msg.confidence * 100)}%
                      </div>
                      <span className="meta-item">⏱ {msg.latency_ms}ms</span>
                      <span className="meta-item">📄 {msg.chunks_retrieved} chunks</span>
                    </div>
                  )}
                </div>
              </div>
            ))}

            {loading && (
              <div className="message assistant">
                <div className="message-avatar">🤖</div>
                <div className="message-content">
                  <div className="message-bubble loading-bubble">
                    <div className="loading-steps">
                      <div className="loading-step active">
                        <div className="pulse"></div>
                        <span>Decomposing query → Retrieving from 15K chunks → Generating expert answer...</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <form className="input-area" onSubmit={handleSubmit}>
            <div className="input-wrapper">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about 3GPP specifications... (e.g. 'What is conditional handover in NR?')"
                disabled={loading}
              />
              <button type="submit" disabled={loading || !input.trim()}>
                {loading ? (
                  <span className="btn-loading">⏳</span>
                ) : (
                  <span>Send</span>
                )}
              </button>
            </div>
            <p className="input-hint">Grounded in official 3GPP specs • Zero hallucination • Exact clause citations</p>
          </form>
        </main>
      </div>
    </div>
  )
}

export default App
