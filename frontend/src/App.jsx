
import { useEffect, useState } from 'react'
import './App.css'

const TEXT_API_URL = (import.meta.env.VITE_TEXT_API_URL || import.meta.env.VITE_API_BASE_URL || 'https://threatguard-ai-docker.onrender.com').replace(/\/$/, '')
const AUDIO_API_URL = (import.meta.env.VITE_AUDIO_API_URL || TEXT_API_URL).replace(/\/$/, '')

async function readApiResponse(response) {
  const contentType = response.headers.get('content-type') || ''

  if (contentType.includes('application/json')) {
    return await response.json()
  }

  const body = await response.text()

  throw new Error(
    `Server returned a non-JSON response (HTTP ${response.status}). ` +
      (response.status >= 500
        ? 'The service may be unavailable or restarting. Please try again shortly.'
        : body.slice(0, 160) || 'Unexpected server response.'),
  )
}
function humanize(key) {
  return String(key)
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function percentage(value) {
  const number = Number(value)
  if (!Number.isFinite(number)) return 'N/A'
  return `${(number * 100).toFixed(2)}%`
}

function scoreEntries(result) {
  if (result.class_probabilities) {
    return Object.entries(result.class_probabilities)
  }

  const candidates = [
    ['Phishing', result.phishing_probability],
    ['Legitimate', result.legitimate_probability],
  ]

  return candidates.filter(([, value]) => value !== undefined && value !== null)
}

function ResultPanel({ result }) {
  if (!result) return null

  const prediction = String(
    result.prediction ?? result.label ?? result.predicted_label ?? 'Analysis complete',
  ).toUpperCase()

  const isPositive = ['PHISHING', 'SPOOF'].includes(prediction)
  const explanation = result.explanation
  const scores = scoreEntries(result)

  const excluded = [
    'warning',
    'input_type',
    'filename',
    'prediction',
    'predicted_label',
    'label',
    'phishing_probability',
    'legitimate_probability',
    'class_probabilities',
    'explanation',
    'model',
    'sample_rate',
    'duration_seconds',
    'embedding_dimensions',
  ]

  const details = Object.entries(result).filter(
    ([key, value]) => !excluded.includes(key) && value !== undefined,
  )

  return (
    <section className="result-panel" aria-live="polite">
      <div className="result-heading">
        <div>
          <span className="eyebrow">ANALYSIS RESULT</span>
          <h3>{humanize(prediction)}</h3>
          <p className="result-subtitle">
            {result.input_type === 'audio'
              ? 'Audio authenticity screening'
              : 'Message threat screening'}
          </p>
        </div>
        <span className={`result-icon ${isPositive ? 'result-icon-alert' : 'result-icon-safe'}`}>
          {isPositive ? '!' : '?'}
        </span>
      </div>

      {result.filename && (
        <p className="file-name">File: {result.filename}</p>
      )}

      {scores.length > 0 && (
        <section className="score-section" aria-label="Model scores">
          <h4>Model scores</h4>
          <p className="section-description">
            Scores returned by the classifier. They are estimates, not guarantees.
          </p>

          {scores.map(([key, value]) => {
            const numericValue = Number(value)
            const valid = Number.isFinite(numericValue)
            const width = valid
              ? Math.min(100, Math.max(0, numericValue * 100))
              : 0

            return (
              <div className="score-row" key={key}>
                <div className="score-label">
                  <span>{humanize(key)}</span>
                  <strong>{percentage(value)}</strong>
                </div>
                <div
                  className="score-track"
                  role="progressbar"
                  aria-label={`${humanize(key)} score`}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={valid ? width : 0}
                >
                  <div className="score-fill" style={{ width: `${width}%` }} />
                </div>
              </div>
            )
          })}
        </section>
      )}

      {explanation && (
        <section className="explanation-section">
          <h4>Why this result?</h4>
          <p className="explanation-summary">{explanation.summary}</p>

          {Array.isArray(explanation.indicators) &&
            explanation.indicators.length > 0 && (
              <div className="indicator-list">
                {explanation.indicators.map((indicator, index) => (
                  <article className="indicator-item" key={`${indicator.title}-${index}`}>
                    <span className="indicator-marker" aria-hidden="true">?</span>
                    <div>
                      <strong>{indicator.title}</strong>
                      <p>{indicator.detail}</p>
                    </div>
                  </article>
                ))}
              </div>
            )}

          {Array.isArray(explanation.indicators) &&
            explanation.indicators.length === 0 && (
              <p className="section-description">
                No configured text indicators were found. That does not mean the
                message is safe.
              </p>
            )}

          {explanation.method && (
            <p className="method-note">
              <strong>Method:</strong> {explanation.method}
            </p>
          )}

          {explanation.limitations && (
            <div className="limitation-note">
              <strong>Important limitations</strong>
              <p>{explanation.limitations}</p>
            </div>
          )}
        </section>
      )}

      {details.length > 0 && (
        <div className="result-details">
          {details.map(([key, value]) => (
            <div className="detail-item" key={key}>
              <span>{humanize(key)}</span>
              <strong>
                {typeof value === 'object' && value !== null ? (
                  <pre>{JSON.stringify(value, null, 2)}</pre>
                ) : (
                  String(value)
                )}
              </strong>
            </div>
          ))}
        </div>
      )}

      {result.warning && <p className="warning">{result.warning}</p>}
    </section>
  )
}

function App() {
  const [mode, setMode] = useState('text')
  const [text, setText] = useState('')
  const [audio, setAudio] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [apiStatus, setApiStatus] = useState('checking')

  useEffect(() => {
    let active = true
    let inFlight = false
    let controller = null

    async function checkHealth() {
      if (!active || inFlight) return

      inFlight = true
      const requestController = new AbortController()
      controller = requestController

      const timeoutId = setTimeout(() => {
        requestController.abort()
      }, 8000)

      try {
        const response = await fetch(`${TEXT_API_URL}/health`, {
          method: 'GET',
          cache: 'no-store',
          signal: requestController.signal,
        })

        if (!response.ok) {
          throw new Error('Health check failed')
        }

        // Accept a successful health endpoint response without assuming
        // whether the server returns JSON or plain text.
        if (active) setApiStatus('online')
      } catch {
        if (active) setApiStatus('offline')
      } finally {
        clearTimeout(timeoutId)
        if (controller === requestController) controller = null
        inFlight = false
      }
    }

    checkHealth()

    // Retry periodically so the UI can recover when the server wakes up.
    const intervalId = setInterval(checkHealth, 10000)

    return () => {
      active = false
      clearInterval(intervalId)
      controller?.abort()
    }
  }, [])

  function changeMode(nextMode) {
    setMode(nextMode)
    setResult(null)
    setError('')
  }

  async function handleTextSubmit(event) {
    event.preventDefault()
    setError('')
    setResult(null)

    if (!text.trim()) {
      setError('Please enter an email or message to analyze.')
      return
    }

    setLoading(true)

    try {
      const response = await fetch(`${TEXT_API_URL}/analyze/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim() }),
      })

      const data = await readApiResponse(response)

      if (!response.ok) {
        throw new Error(data.detail || 'Text analysis failed.')
      }

      setResult(data)
    } catch (err) {
      setError(
        err.message === 'Failed to fetch'
          ? 'Unable to reach the service. It may be waking up or temporarily unavailable. Please try again shortly.'
          : err.message,
      )
    } finally {
      setLoading(false)
    }
  }

  async function handleAudioSubmit(event) {
    event.preventDefault()
    setError('')
    setResult(null)

    if (!audio) {
      setError('Please choose an audio file first.')
      return
    }

    setLoading(true)

    try {
      const formData = new FormData()
      formData.append('file', audio)

      const response = await fetch(`${AUDIO_API_URL}/analyze/audio`, {
        method: 'POST',
        body: formData,
      })

      const data = await readApiResponse(response)

      if (!response.ok) {
        throw new Error(data.detail || 'Audio analysis failed.')
      }

      setResult(data)
    } catch (err) {
      setError(
        err.message === 'Failed to fetch'
          ? 'Unable to reach the service. It may be waking up or temporarily unavailable. Please try again shortly.'
          : err.message,
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#home">
          <span className="brand-mark">T</span>
          <span>
            ThreatGuard<span className="brand-ai"> AI</span>
          </span>
        </a>

        <div className={`api-status ${apiStatus}`} role="status" aria-live="polite">
          <span className="status-dot" />
          API {apiStatus === 'checking' ? 'Checking' : apiStatus}
        </div>
      </header>

      <main id="home">
        <section className="hero">
          <span className="eyebrow">AI-POWERED SECURITY LAB</span>
          <h1>
            Detect threats.
            <br />
            <span>Build digital trust.</span>
          </h1>
          <p>
            Analyze suspicious messages and potentially synthetic audio using
            machine-learning models in one workspace.
          </p>

          <div className="hero-tags">
            <span>Text intelligence</span>
            <span>Audio analysis</span>
            <span>Model-driven insights</span>
          </div>
        </section>

        <section className="workspace">
          <div className="section-heading">
            <div>
              <span className="eyebrow">ANALYSIS WORKSPACE</span>
              <h2>Scan a potential threat</h2>
            </div>
            <span className="workspace-number">01 / ANALYZE</span>
          </div>

          <div className="mode-tabs" role="tablist" aria-label="Analysis type">
            <button
              type="button"
              role="tab"
              aria-selected={mode === 'text'}
              className={mode === 'text' ? 'mode-tab active' : 'mode-tab'}
              onClick={() => changeMode('text')}
            >
              <span className="tab-icon" aria-hidden="true">T</span>
              <span>
                <strong>Text scanner</strong>
                <small>Emails &amp; messages</small>
              </span>
            </button>

            <button
              type="button"
              role="tab"
              aria-selected={mode === 'audio'}
              className={mode === 'audio' ? 'mode-tab active' : 'mode-tab'}
              onClick={() => changeMode('audio')}
            >
              <span className="tab-icon" aria-hidden="true">A</span>
              <span>
                <strong>Audio scanner</strong>
                <small>Voice &amp; recordings</small>
              </span>
            </button>
          </div>

          {mode === 'text' ? (
            <form className="scanner-card" onSubmit={handleTextSubmit}>
              <label htmlFor="message">Email or message content</label>
              <p className="field-hint">
                Paste the text you want the model to examine.
              </p>

              <textarea
                id="message"
                value={text}
                onChange={(event) => setText(event.target.value)}
                placeholder="Example: Your account will be suspended. Verify your details immediately..."
                maxLength={20000}
                rows={8}
                disabled={loading}
              />

              <div className="input-footer">
                <span>{text.length.toLocaleString()} / 20,000 characters</span>
                <span>Text model â€” DistilBERT</span>
              </div>

              <button className="primary-button" type="submit" disabled={loading}>
                {loading ? 'Analyzing text...' : 'Analyze message'}
              </button>
            </form>
          ) : (
            <form className="scanner-card" onSubmit={handleAudioSubmit}>
              <label htmlFor="audio-file">Audio recording</label>
              <p className="field-hint">
                Select a recording to check for potential synthetic audio.
              </p>

              <label className="upload-zone" htmlFor="audio-file">
                <span className="upload-icon" aria-hidden="true">â†‘</span>
                <strong>{audio ? audio.name : 'Choose an audio file'}</strong>
                <span>
                  {audio
                    ? `${(audio.size / (1024 * 1024)).toFixed(2)} MB`
                    : 'WAV, MP3, FLAC or OGG â€” Maximum 20 MB'}
                </span>
                <input
                  id="audio-file"
                  type="file"
                  accept=".wav,.mp3,.flac,.ogg,audio/wav,audio/mpeg,audio/flac,audio/ogg"
                  onChange={(event) => {
                    setAudio(event.target.files?.[0] ?? null)
                    setResult(null)
                    setError('')
                  }}
                  disabled={loading}
                />
              </label>

              <div className="input-footer">
                <span>Audio model â€” Wav2Vec2 embeddings</span>
                <span>Experimental classifier</span>
              </div>

              <button className="primary-button" type="submit" disabled={loading}>
                {loading ? 'Analyzing audio...' : 'Analyze recording'}
              </button>
            </form>
          )}

          {error && (
            <div className="error-message" role="alert">
              <strong>Analysis could not be completed</strong>
              <p>{error}</p>
            </div>
          )}

          {result && <ResultPanel result={result} />}
        </section>

        <section className="trust-note">
          <span className="trust-icon" aria-hidden="true">i</span>
          <div>
            <strong>Use predictions as signals, not proof.</strong>
            <p>
              These models are experimental. A prediction can be incorrect,
              and a high model score is not a guarantee. Verify suspicious
              messages or recordings independently.
            </p>
          </div>
        </section>
      </main>

      <footer className="footer">
        <span>THREATGUARD AI</span>
        <span>Experimental research project â€” 2026</span>
      </footer>
    </div>
  )
}

export default App





