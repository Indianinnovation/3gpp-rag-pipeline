import React, { useState, useRef, useEffect, useCallback, useReducer } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const SAMPLE_QUERIES = [
  "What are the RRC states in 5G NR?",
  "Explain handover procedure in NR",
  "What is carrier aggregation in NR?",
  "How does HARQ work in 5G?",
  "Clear code classification in 5G NR",
  "Describe the RRC connection setup procedure"
]

const API_URL = 'https://api.ask3gpp.com'

const VALID_USER = 'admin'
const VALID_PASS = 'admin@3gpp'

// ─── SSE State Machine ───────────────────────────────────────────────────────
const STEP_ORDER = ['planner', 'router', 'isrel', 'crag', 'generator']

const initialStreamState = {
  status: 'idle', // idle | streaming | done
  steps: {},
  tokens: '',
  sources: [],
  confidence: null,
  totalMs: null,
  chunksUsed: null,
}

function streamReducer(state, action) {
  switch (action.type) {
    case 'RESET':
      return { ...initialStreamState, status: 'streaming' }
    case 'STEP_START':
      return {
        ...state,
        steps: {
          ...state.steps,
          [action.step]: { status: 'running', label: action.label, icon: action.icon, detail: action.detail, ms: null }
        }
      }
    case 'STEP_DONE':
      return {
        ...state,
        steps: {
          ...state.steps,
          [action.step]: { ...state.steps[action.step], status: 'done', label: action.label, ms: action.ms, sourcesPreview: action.sourcesPreview, verdict: action.verdict }
        }
      }
    case 'TOKEN':
      return { ...state, tokens: state.tokens + action.text }
    case 'DONE':
      return { ...state, status: 'done', sources: action.sources, confidence: action.confidence, totalMs: action.total_ms, chunksUsed: action.chunks_used }
    case 'ERROR':
      return { ...state, status: 'done' }
    default:
      return state
  }
}

// ─── SSE Client Hook ─────────────────────────────────────────────────────────
function useQueryStream() {
  const [streamState, dispatch] = useReducer(streamReducer, initialStreamState)

  const query = useCallback(async (text, specFilter, releaseFilter) => {
    dispatch({ type: 'RESET' })

    try {
      const res = await fetch(`${API_URL}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text, spec_filter: specFilter || null, release_filter: releaseFilter || null })
      })

      if (!res.ok) {
        // Fallback to non-streaming endpoint
        const fallbackRes = await fetch(`${API_URL}/query`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: text, spec_filter: specFilter || null, release_filter: releaseFilter || null })
        })
        const data = await fallbackRes.json()
        // Simulate steps from response
        if (data.steps) {
          for (const step of data.steps) {
            dispatch({ type: 'STEP_DONE', step: step.name.split(':')[0].toLowerCase().replace(/[^a-z]/g, ''), label: step.name, ms: step.ms })
          }
        }
        dispatch({ type: 'TOKEN', text: data.answer })
        dispatch({ type: 'DONE', sources: data.citations || [], confidence: data.confidence, total_ms: data.latency_ms, chunks_used: data.chunks_retrieved })
        return
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n\n')
        buffer = lines.pop() || ''
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event = JSON.parse(line.slice(6))
              switch (event.type) {
                case 'step_start':
                  dispatch({ type: 'STEP_START', step: event.step, label: event.label, icon: event.icon, detail: event.detail })
                  break
                case 'step_done':
                  dispatch({ type: 'STEP_DONE', step: event.step, label: event.label, ms: event.ms, sourcesPreview: event.sources_preview, verdict: event.verdict })
                  break
                case 'token':
                  dispatch({ type: 'TOKEN', text: event.text })
                  break
                case 'done':
                  dispatch({ type: 'DONE', sources: event.sources, confidence: event.confidence, total_ms: event.total_ms, chunks_used: event.chunks_used })
                  break
              }
            } catch (e) { /* skip malformed events */ }
          }
        }
      }
    } catch (err) {
      dispatch({ type: 'TOKEN', text: `**Error:** ${err.message}\n\nMake sure the backend is running.` })
      dispatch({ type: 'ERROR' })
    }
  }, [])

  return { streamState, query }
}

// ─── Pipeline Steps Component ────────────────────────────────────────────────
function PipelineSteps({ steps, status, totalMs, sources }) {
  const [collapsed, setCollapsed] = useState(false)

  const stepEntries = STEP_ORDER.map(id => ({ id, ...(steps[id] || { status: 'pending' }) }))
  const completedCount = stepEntries.filter(s => s.status === 'done').length
  const specList = (sources || []).slice(0, 3).map(s => `TS ${s.spec}`).join(', ')
  const moreCount = (sources || []).length - 3

  if (status === 'idle') return null

  return (
    <div className="pipeline-panel">
      <button className="pipeline-toggle" onClick={() => setCollapsed(!collapsed)}>
        <span className="toggle-arrow">{collapsed ? '▶' : '▼'}</span>
        <span className="toggle-label">
          {status === 'done'
            ? `${completedCount} steps · ${totalMs ? (totalMs / 1000).toFixed(1) + 's' : ''} · ${specList}${moreCount > 0 ? ` (+${moreCount})` : ''}`
            : `Pipeline · running...`
          }
        </span>
      </button>

      {!collapsed && (
        <div className="pipeline-steps-list">
          {stepEntries.map(step => (
            <div key={step.id} className={`pipeline-step ${step.status || 'pending'}`}>
              <span className="step-status-icon">
                {step.status === 'done' ? '✓' : step.status === 'running' ? <span className="pulse-dot" /> : '○'}
              </span>
              <span className="step-icon">{step.icon || '·'}</span>
              <span className="step-label">{step.label || step.id}</span>
              {step.ms != null && <span className="step-ms">{step.ms.toLocaleString()}ms</span>}
              {step.verdict && <span className={`step-verdict ${step.verdict}`}>{step.verdict}</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ─── Sources Panel ───────────────────────────────────────────────────────────
const SPEC_COLORS = {
  '38331': '#3b82f6', '24501': '#8b5cf6', '38413': '#f97316',
  '38473': '#10b981', '38423': '#14b8a6', '38463': '#ec4899',
  '38.413': '#f97316', '38.473': '#10b981', '38.423': '#14b8a6',
}

function SourcesPanel({ sources }) {
  if (!sources || sources.length === 0) return null
  return (
    <div className="sources-panel">
      <div className="sources-header">📚 Sources ({sources.length})</div>
      <div className="sources-grid">
        {sources.map((s, i) => (
          <div key={i} className="source-card" style={{ borderLeftColor: SPEC_COLORS[s.spec] || '#64748b' }}>
            <div className="source-spec">TS {s.spec}</div>
            <div className="source-clause">§{s.clause}</div>
            <div className="source-meta">
              <span className="source-release">{s.release}</span>
              <span className="source-score">{(s.score * 100).toFixed(1)}%</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── Confidence Badge ────────────────────────────────────────────────────────
function ConfidenceBadge({ confidence, totalMs, chunksUsed }) {
  if (confidence == null) return null
  const pct = Math.round(confidence * 100)
  const color = pct >= 70 ? '#10b981' : pct >= 40 ? '#f59e0b' : '#ef4444'
  return (
    <div className="confidence-bar">
      <div className="confidence-gauge">
        <div className="confidence-fill" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="confidence-text" style={{ color }}>Confidence: {pct}%</span>
      {totalMs && <span className="meta-stat">⏱ {(totalMs / 1000).toFixed(1)}s</span>}
      {chunksUsed && <span className="meta-stat">📄 {chunksUsed} chunks</span>}
    </div>
  )
}

// ─── Login ───────────────────────────────────────────────────────────────────
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
          <h1>ask3gpp</h1>
          <p>AI-Powered 3GPP Specification Search</p>
        </div>
        <form onSubmit={handleLogin}>
          <div className="login-field">
            <label>Username</label>
            <input type="text" value={username} onChange={(e) => { setUsername(e.target.value); setError('') }} placeholder="Enter username" autoFocus />
          </div>
          <div className="login-field">
            <label>Password</label>
            <input type="password" value={password} onChange={(e) => { setPassword(e.target.value); setError('') }} placeholder="Enter password" />
          </div>
          {error && <p className="login-error">{error}</p>}
          <button type="submit" className="login-btn">Sign In</button>
        </form>
        <p className="login-footer">Grounded in official 3GPP specs • Zero hallucination • Exact clause citations</p>
      </div>
    </div>
  )
}

// ─── Main App ────────────────────────────────────────────────────────────────
function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [specFilter, setSpecFilter] = useState('')
  const [releaseFilter, setReleaseFilter] = useState('')
  const { streamState, query: streamQuery } = useQueryStream()
  const messagesEndRef = useRef(null)
  const isStreaming = streamState.status === 'streaming'

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamState.tokens])

  // When streaming completes, save to messages
  useEffect(() => {
    if (streamState.status === 'done' && streamState.tokens) {
      // Clean confidence JSON from answer
      let answer = streamState.tokens
      const jsonMatch = answer.match(/\{[^{}]*"confidence"[^{}]*\}/)
      if (jsonMatch) {
        answer = answer.slice(0, answer.indexOf(jsonMatch[0])).trim()
      }

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: answer,
        sources: streamState.sources,
        confidence: streamState.confidence,
        totalMs: streamState.totalMs,
        chunksUsed: streamState.chunksUsed,
        steps: streamState.steps,
        timestamp: new Date()
      }])
    }
  }, [streamState.status])

  const sendQuery = async (queryText) => {
    if (!queryText.trim() || isStreaming) return
    setMessages(prev => [...prev, { role: 'user', content: queryText, timestamp: new Date() }])
    setInput('')
    streamQuery(queryText, specFilter, releaseFilter)
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    sendQuery(input)
  }

  if (!isAuthenticated) {
    return <LoginPage onLogin={setIsAuthenticated} />
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <span className="logo-icon">📡</span>
            <h1>ask3gpp</h1>
            <span className="badge">Powered by Amazon Bedrock</span>
          </div>
          <div className="header-right">
            <span className="status-badge">● 40K+ chunks indexed</span>
            <button className="clear-btn" onClick={() => setMessages([])}>Clear</button>
          </div>
        </div>
      </header>

      <div className="main-container">
        <aside className="sidebar">
          <div className="sidebar-section">
            <h3>🔍 Filters</h3>
            <label>Spec Number</label>
            <input type="text" placeholder="e.g. 38331" value={specFilter} onChange={(e) => setSpecFilter(e.target.value)} />
            <label>Release</label>
            <input type="text" placeholder="e.g. Rel-18" value={releaseFilter} onChange={(e) => setReleaseFilter(e.target.value)} />
          </div>

          <div className="sidebar-section">
            <h3>⚡ Quick Queries</h3>
            {SAMPLE_QUERIES.map((q, i) => (
              <button key={i} className="sample-btn" onClick={() => sendQuery(q)} disabled={isStreaming}>
                <span className="sample-icon">→</span> {q}
              </button>
            ))}
          </div>
        </aside>

        <main className="chat-area">
          <div className="messages">
            {messages.length === 0 && !isStreaming && (
              <div className="welcome">
                <div className="welcome-icon">📡</div>
                <h2>3GPP Specification Expert</h2>
                <p>Ask any question about 5G NR, LTE, or 3GPP standards. Every answer is grounded in official specification text with exact clause citations.</p>
              </div>
            )}

            {messages.map((msg, i) => (
              <div key={i} className={`message ${msg.role}`}>
                <div className="message-avatar">{msg.role === 'user' ? '👤' : '🤖'}</div>
                <div className="message-content">
                  {msg.role === 'user' ? (
                    <p className="user-text">{msg.content}</p>
                  ) : (
                    <>
                      {msg.steps && Object.keys(msg.steps).length > 0 && (
                        <PipelineSteps steps={msg.steps} status="done" totalMs={msg.totalMs} sources={msg.sources} />
                      )}
                      <div className="markdown-body">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                      </div>
                      <SourcesPanel sources={msg.sources} />
                      <ConfidenceBadge confidence={msg.confidence} totalMs={msg.totalMs} chunksUsed={msg.chunksUsed} />
                    </>
                  )}
                </div>
              </div>
            ))}

            {/* Live streaming message */}
            {isStreaming && (
              <div className="message assistant">
                <div className="message-avatar">🤖</div>
                <div className="message-content">
                  <PipelineSteps steps={streamState.steps} status="streaming" />
                  {streamState.tokens && (
                    <div className="markdown-body streaming">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{streamState.tokens}</ReactMarkdown>
                      <span className="streaming-cursor">▊</span>
                    </div>
                  )}
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
                disabled={isStreaming}
              />
              <button type="submit" disabled={isStreaming || !input.trim()}>
                {isStreaming ? '⏳' : 'Send'}
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
