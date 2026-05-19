import React, { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'

const SAMPLE_QUERIES = [
  "What are the RRC states in 5G NR?",
  "Explain handover procedure in NR",
  "What is carrier aggregation in NR?",
  "How does HARQ work in 5G?",
  "What is the role of gNB-DU and gNB-CU?"
]

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [specFilter, setSpecFilter] = useState('')
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
          spec_filter: specFilter || null
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
        content: `Error: ${err.message}. Make sure the backend is running on port 8000.`,
        timestamp: new Date()
      }])
    }
    setLoading(false)
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    sendQuery(input)
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <span className="logo-icon">📡</span>
            <h1>3GPP RAG</h1>
          </div>
          <p className="subtitle">AI-Powered 3GPP Specification Search • Powered by Amazon Bedrock</p>
        </div>
      </header>

      <div className="main-container">
        <aside className="sidebar">
          <div className="sidebar-section">
            <h3>Filters</h3>
            <label>Spec Number</label>
            <input
              type="text"
              placeholder="e.g. 38331"
              value={specFilter}
              onChange={(e) => setSpecFilter(e.target.value)}
            />
          </div>
          <div className="sidebar-section">
            <h3>Sample Queries</h3>
            {SAMPLE_QUERIES.map((q, i) => (
              <button key={i} className="sample-btn" onClick={() => sendQuery(q)}>
                {q}
              </button>
            ))}
          </div>
          <div className="sidebar-section">
            <h3>Architecture</h3>
            <div className="arch-info">
              <span>🔍 Hybrid Search</span>
              <span>🧠 Claude Sonnet 4.5</span>
              <span>📊 pgvector + HNSW</span>
              <span>🔗 LangGraph Engine</span>
            </div>
          </div>
        </aside>

        <main className="chat-area">
          <div className="messages">
            {messages.length === 0 && (
              <div className="welcome">
                <h2>Ask anything about 3GPP specifications</h2>
                <p>This RAG pipeline searches across 13,000+ chunks from NR specs (TS 38.xxx) and RAN2 meeting documents.</p>
              </div>
            )}
            {messages.map((msg, i) => (
              <div key={i} className={`message ${msg.role}`}>
                <div className="message-bubble">
                  {msg.role === 'assistant' ? (
                    <>
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                      {msg.citations && msg.citations.length > 0 && (
                        <div className="citations">
                          <h4>📚 Citations</h4>
                          {msg.citations.map((c, j) => (
                            <span key={j} className="citation-tag">
                              TS {c.spec} §{c.section} | {c.release} ({c.score})
                            </span>
                          ))}
                        </div>
                      )}
                      {msg.confidence !== undefined && (
                        <div className="meta">
                          <span className={`confidence ${msg.confidence >= 0.7 ? 'high' : msg.confidence >= 0.4 ? 'medium' : 'low'}`}>
                            Confidence: {Math.round(msg.confidence * 100)}%
                          </span>
                          <span>⏱ {msg.latency_ms}ms</span>
                          <span>📄 {msg.chunks_retrieved} chunks</span>
                        </div>
                      )}
                    </>
                  ) : (
                    <p>{msg.content}</p>
                  )}
                </div>
              </div>
            ))}
            {loading && (
              <div className="message assistant">
                <div className="message-bubble loading-bubble">
                  <div className="typing-indicator">
                    <span></span><span></span><span></span>
                  </div>
                  <p>Searching specs & generating answer...</p>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <form className="input-area" onSubmit={handleSubmit}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about 3GPP specifications..."
              disabled={loading}
            />
            <button type="submit" disabled={loading || !input.trim()}>
              {loading ? '⏳' : '🔍'}
            </button>
          </form>
        </main>
      </div>
    </div>
  )
}

export default App
