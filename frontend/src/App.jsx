import React, { useState, useRef, useEffect, useCallback } from 'react'
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

const API_URL = 'https://api.ask3gpp.com'  // ECS Fargate backend

const VALID_USER = 'admin'
const VALID_PASS = 'admin@3gpp'

// Streaming text hook
function useStreamingText(text, speed = 8) {
  const [displayed, setDisplayed] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const indexRef = useRef(0)
  const intervalRef = useRef(null)

  useEffect(() => {
    if (!text) { setDisplayed(''); return }
    setIsStreaming(true)
    indexRef.current = 0
    setDisplayed('')

    intervalRef.current = setInterval(() => {
      indexRef.current += speed
      if (indexRef.current >= text.length) {
        setDisplayed(text)
        setIsStreaming(false)
        clearInterval(intervalRef.current)
      } else {
        setDisplayed(text.slice(0, indexRef.current))
      }
    }, 16)

    return () => clearInterval(intervalRef.current)
  }, [text, speed])

  const skipToEnd = useCallback(() => {
    clearInterval(intervalRef.current)
    setDisplayed(text)
    setIsStreaming(false)
  }, [text])

  return { displayed, isStreaming, skipToEnd }
}

function StreamingMessage({ content, onStreamEnd }) {
  const { displayed, isStreaming, skipToEnd } = useStreamingText(content, 12)

  useEffect(() => {
    if (!isStreaming && content && onStreamEnd) onStreamEnd()
  }, [isStreaming])

  return (
    <div className="markdown-body">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {displayed}
      </ReactMarkdown>
      {isStreaming && <span className="streaming-cursor">▊</span>}
      {isStreaming && <button className="skip-btn" onClick={skipToEnd}>Skip →</button>}
    </div>
  )
}

// Login component
function LoginPage({ onLogin }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')

  const handleLogin = (e) => {
    e.preventDefault()
    if (username === VALID_USER && password === VALID_PASS) {
      onLogin(true)
    } else {
      setError('Invalid credentials')
    }
  }

  return (
    <div className="login-page">
      <div className="login-card">
        <div className="login-header">
          <span className="login-icon">📡</span>
          <h1>3GPP RAG Expert</h1>
          <p>AI-Powered 3GPP Specification Search</p>
        </div>
        <form onSubmit={handleLogin}>
          <div className="login-field">
            <label>Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => { setUsername(e.target.value); setError('') }}
              placeholder="Enter username"
              autoFocus
            />
          </div>
          <div className="login-field">
            <label>Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => { setPassword(e.target.value); setError('') }}
              placeholder="Enter password"
            />
          </div>
          {error && <p className="login-error">{error}</p>}
          <button type="submit" className="login-btn">Sign In</button>
        </form>
        <p className="login-footer">Powered by Amazon Bedrock • pgvector • LangGraph</p>
      </div>
    </div>
  )
}

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [specFilter, setSpecFilter] = useState('')
  const [releaseFilter, setReleaseFilter] = useState('')
  const [streamingIdx, setStreamingIdx] = useState(-1)
  const [liveSteps, setLiveSteps] = useState([])
  const [liveTokens, setLiveTokens] = useState('')
  const [isStreamActive, setIsStreamActive] = useState(false)
  const messagesEndRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingIdx, liveTokens, liveSteps])

  const sendQuery = async (query) => {
    if (!query.trim()) return
    const userMsg = { role: 'user', content: query, timestamp: new Date() }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)
    setLiveSteps([])
    setLiveTokens('')
    setIsStreamActive(true)

    try {
      // Try SSE streaming endpoint first
      const res = await fetch(`${API_URL}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          spec_filter: specFilter || null,
          release_filter: releaseFilter || null
        })
      })

      if (!res.ok || !res.body) {
        // Fallback to non-streaming
        const fallbackRes = await fetch(`${API_URL}/query`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, spec_filter: specFilter || null, release_filter: releaseFilter || null })
        })
        const data = await fallbackRes.json()
        setIsStreamActive(false)
        const assistantMsg = {
          role: 'assistant', content: data.answer, citations: data.citations,
          confidence: data.confidence, latency_ms: data.latency_ms,
          chunks_retrieved: data.chunks_retrieved, steps: data.steps || [],
          cached: data.cached || false, timestamp: new Date(), isNew: true
        }
        setMessages(prev => { const n = [...prev, assistantMsg]; setStreamingIdx(n.length - 1); return n })
        setLoading(false)
        return
      }

      // SSE streaming
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let allTokens = ''
      let allSteps = []
      let finalSources = []
      let finalConfidence = 0.5
      let finalMs = 0
      let finalChunks = 0

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n\n')
        buffer = lines.pop() || ''
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const event = JSON.parse(line.slice(6))
            switch (event.type) {
              case 'step_start':
                allSteps = [...allSteps, { name: event.label, icon: event.icon, status: 'running', detail: event.detail }]
                setLiveSteps([...allSteps])
                break
              case 'step_done':
                allSteps = allSteps.map(s => s.name === event.label || s.status === 'running'
                  ? { ...s, name: event.label, status: 'done', ms: event.ms, detail: event.label }
                  : s)
                // Mark last running step as done
                const lastRunning = allSteps.findLastIndex(s => s.status === 'running')
                if (lastRunning >= 0) allSteps[lastRunning] = { ...allSteps[lastRunning], name: event.label, status: 'done', ms: event.ms }
                setLiveSteps([...allSteps])
                break
              case 'token':
                allTokens += event.text
                setLiveTokens(allTokens)
                break
              case 'done':
                finalSources = event.sources || []
                finalConfidence = event.confidence
                finalMs = event.total_ms
                finalChunks = event.chunks_used
                break
            }
          } catch (e) { /* skip malformed */ }
        }
      }

      // Stream complete — save to messages
      setIsStreamActive(false)
      let answer = allTokens
      const jsonMatch = answer.match(/\{[^{}]*"confidence"[^{}]*\}/)
      if (jsonMatch) answer = answer.slice(0, answer.indexOf(jsonMatch[0])).trim()

      const assistantMsg = {
        role: 'assistant', content: answer,
        citations: finalSources.map(s => ({ spec: s.spec, section: s.clause, release: s.release, score: s.score })),
        confidence: finalConfidence, latency_ms: finalMs, chunks_retrieved: finalChunks,
        steps: allSteps.map(s => ({ name: s.name, status: s.status, ms: s.ms })),
        cached: false, timestamp: new Date(), isNew: true
      }
      setMessages(prev => { const n = [...prev, assistantMsg]; setStreamingIdx(n.length - 1); return n })
      setLiveSteps([])
      setLiveTokens('')

    } catch (err) {
      setIsStreamActive(false)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `**Error:** ${err.message}\n\nMake sure the backend is running:\n\`\`\`bash\ncd backend && uvicorn api:app --reload --port 8000\n\`\`\``,
        timestamp: new Date(), isNew: true
      }])
    }
    setLoading(false)
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    sendQuery(input)
  }

  const clearChat = () => { setMessages([]); setStreamingIdx(-1) }

  if (!isAuthenticated) {
    return <LoginPage onLogin={setIsAuthenticated} />
  }

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
            <span className="status-badge">● 40K+ chunks indexed</span>
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

          <div className="sidebar-section indexed-docs">
            <h3>📂 Indexed Sources</h3>
            <div className="indexed-list">
              <div className="indexed-group">
                <span className="indexed-badge meeting">Meetings</span>
                <ul>
                  <li>TSGR2_129 Docs + LS</li>
                  <li>TSGR2_129bis Docs + LS</li>
                  <li>TSGR_109 (RAN Plenary)</li>
                </ul>
              </div>
              <div className="indexed-group">
                <span className="indexed-badge specs">Specifications</span>
                <ul>
                  <li>Rel-18 / 38-series (NR)</li>
                  <li>Rel-19 / 38-series (NR)</li>
                  <li>Rel-20 / 38-series (NR)</li>
                </ul>
              </div>
              <div className="indexed-group">
                <span className="indexed-badge stats">Stats</span>
                <ul>
                  <li>40,000+ chunks indexed</li>
                  <li>140+ spec files parsed</li>
                  <li>Hybrid: vector + keyword</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="sidebar-section">
            <h3>⚡ Quick Queries</h3>
            {SAMPLE_QUERIES.map((q, i) => (
              <button key={i} className="sample-btn" onClick={() => sendQuery(q)} disabled={loading}>
                <span className="sample-icon">→</span> {q}
              </button>
            ))}
          </div>

          <div className="sidebar-section">
            <h3>🏗️ Pipeline</h3>
            <div className="pipeline-steps">
              <div className="step"><span className="step-num">1</span><span>Query Decomposition</span></div>
              <div className="step"><span className="step-num">2</span><span>Multi-Query Retrieval</span></div>
              <div className="step"><span className="step-num">3</span><span>Score & Rerank</span></div>
              <div className="step"><span className="step-num">4</span><span>Expert Generation</span></div>
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

          <div className="sidebar-section">
            <h3>📜 Query History</h3>
            <div className="history-list">
              {messages.filter(m => m.role === 'user').length === 0 && (
                <p className="history-empty">No queries yet</p>
              )}
              {messages.filter(m => m.role === 'user').slice(-10).reverse().map((m, i) => (
                <button key={i} className="history-btn" onClick={() => sendQuery(m.content)} disabled={loading}>
                  {m.content.length > 40 ? m.content.slice(0, 40) + '…' : m.content}
                </button>
              ))}
            </div>
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
                  <div className="feature"><span>🎯</span><strong>More Accurate</strong><p>Than ChatGPT/Gemini — grounded in exact spec text</p></div>
                  <div className="feature"><span>📋</span><strong>Structured</strong><p>Tables, diagrams, protocol flows</p></div>
                  <div className="feature"><span>🔗</span><strong>Traceable</strong><p>Every claim linked to TS clause</p></div>
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
                      i === streamingIdx && msg.isNew ? (
                        <StreamingMessage
                          content={msg.content}
                          onStreamEnd={() => {
                            setMessages(prev => prev.map((m, idx) =>
                              idx === i ? { ...m, isNew: false } : m
                            ))
                            setStreamingIdx(-1)
                          }}
                        />
                      ) : (
                        <div className="markdown-body">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {msg.content}
                          </ReactMarkdown>
                        </div>
                      )
                    ) : (
                      <p>{msg.content}</p>
                    )}
                  </div>

                  {msg.role === 'assistant' && !msg.isNew && msg.steps && msg.steps.length > 0 && (
                    <div className="steps-panel">
                      <div className="steps-header">🔄 Pipeline Steps</div>
                      <div className="steps-list">
                        {msg.steps.map((step, j) => (
                          <div key={j} className={`step-item ${step.status}`}>
                            <span className="step-icon">{step.status === 'done' ? '✓' : '⏳'}</span>
                            <span className="step-name">{step.name}</span>
                            {step.ms !== undefined && <span className="step-time">{step.ms}ms</span>}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {msg.role === 'assistant' && !msg.isNew && msg.citations && msg.citations.length > 0 && (
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

                  {msg.role === 'assistant' && !msg.isNew && msg.confidence !== undefined && (
                    <div className="meta-bar">
                      {msg.cached && <span className="cache-badge">⚡ Cached</span>}
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
                    {/* Live Pipeline Steps */}
                    {liveSteps.length > 0 && (
                      <div className="live-pipeline">
                        <div className="pipeline-header">🔄 Pipeline Steps</div>
                        {liveSteps.map((step, j) => (
                          <div key={j} className={`live-step ${step.status}`}>
                            <span className="step-status-icon">
                              {step.status === 'done' ? '✓' : <span className="pulse-dot">⟳</span>}
                            </span>
                            <span className="step-icon">{step.icon}</span>
                            <span className="step-name">{step.name}</span>
                            {step.ms != null && <span className="step-time">{step.ms.toLocaleString()}ms</span>}
                          </div>
                        ))}
                      </div>
                    )}
                    {/* Live Token Stream */}
                    {liveTokens && (
                      <div className="markdown-body">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {liveTokens}
                        </ReactMarkdown>
                        <span className="streaming-cursor">▊</span>
                      </div>
                    )}
                    {/* Initial loading state before first step arrives */}
                    {!liveSteps.length && !liveTokens && (
                      <div className="loading-steps">
                        <div className="loading-step active">
                          <div className="pulse"></div>
                          <span>Connecting to pipeline...</span>
                        </div>
                      </div>
                    )}
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
                {loading ? <span className="btn-loading">⏳</span> : <span>Send</span>}
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
