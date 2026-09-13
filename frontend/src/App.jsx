import React, { useState, useEffect, useRef } from 'react'
import {
  Upload,
  FileText,
  ShieldCheck,
  Cpu,
  AlertCircle,
  CheckCircle2,
  AlertTriangle,
  Play,
  Eye,
  Info,
  Server,
  Layers,
  ArrowRight,
  ExternalLink
} from 'lucide-react'

// Backend API URL from environment variable, defaulting to localhost:8000 for local dev
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const CLASS_COLORS = {
  amount_due: '#10b981',
  date_due: '#f59e0b',
  document_id: '#8b5cf6',
  date_issue: '#0ea5e9',
  vendor_name: '#ec4899',
}

const SAMPLE_REASONING_PRESETS = {
  clear_overdue: {
    document_id: "INV-2026-OVERDUE",
    as_of_date: "2026-09-01",
    fields: [
      { field: "amount_due", value: "4500.00", source: "ocr", confidence: 0.95, bbox: [100, 200, 300, 250] },
      { field: "date_due", value: "2026-08-15", source: "ocr", confidence: 0.92, bbox: [100, 300, 250, 340] },
      { field: "vendor_name", value: "Acme Logistics", source: "ocr", confidence: 0.88, bbox: [50, 50, 250, 90] }
    ]
  },
  upcoming_due: {
    document_id: "INV-2026-UPCOMING",
    as_of_date: "2026-09-01",
    fields: [
      { field: "amount_due", value: "1200.00", source: "ocr", confidence: 0.94, bbox: [100, 200, 300, 250] },
      { field: "date_due", value: "2026-09-20", source: "ocr", confidence: 0.91, bbox: [100, 300, 250, 340] }
    ]
  },
  missing_amount: {
    document_id: "INV-2026-NO-AMOUNT",
    as_of_date: "2026-09-01",
    fields: [
      { field: "date_due", value: "2026-08-20", source: "ocr", confidence: 0.90, bbox: [100, 300, 250, 340] }
    ]
  }
}

export default function App() {
  const [activeTab, setActiveTab] = useState('pipeline')
  const [backendHealth, setBackendHealth] = useState({ status: 'checking', model_loaded: false, service: '' })
  
  // Pipeline state
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [documentId, setDocumentId] = useState('')
  const [asOfDate, setAsOfDate] = useState('2026-09-15')
  const [loading, setLoading] = useState(false)
  const [errorMsg, setErrorMsg] = useState(null)
  const [detectionData, setDetectionData] = useState(null)
  const [processData, setProcessData] = useState(null)

  // Direct reasoning state
  const [reasonJson, setReasonJson] = useState(JSON.stringify(SAMPLE_REASONING_PRESETS.clear_overdue, null, 2))
  const [reasonResult, setReasonResult] = useState(null)
  const [reasonLoading, setReasonLoading] = useState(false)

  const imageRef = useRef(null)
  const canvasRef = useRef(null)

  // Check health on load and periodically
  useEffect(() => {
    checkHealth()
    const interval = setInterval(checkHealth, 15000)
    return () => clearInterval(interval)
  }, [])

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`)
      if (res.ok) {
        const data = await res.json()
        setBackendHealth({ status: 'online', model_loaded: data.model_loaded, service: data.service, path: data.model_path })
      } else {
        setBackendHealth({ status: 'standby', model_loaded: false })
      }
    } catch {
      setBackendHealth({ status: 'offline', model_loaded: false })
    }
  }

  const handleFileSelect = (file) => {
    if (!file) return
    setSelectedFile(file)
    setDetectionData(null)
    setProcessData(null)
    setErrorMsg(null)
    if (!documentId) {
      setDocumentId(file.name.replace(/\.[^/.]+$/, ''))
    }
    if (file.type.startsWith('image/')) {
      const url = URL.createObjectURL(file)
      setPreviewUrl(url)
    } else {
      setPreviewUrl(null)
    }
  }

  // Draw bounding boxes on canvas over the invoice image
  useEffect(() => {
    const canvas = canvasRef.current
    const img = imageRef.current
    if (!canvas || !img || !detectionData) return

    const ctx = canvas.getContext('2d')
    canvas.width = img.clientWidth
    canvas.height = img.clientHeight
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const scaleX = canvas.width / (detectionData.image_width || img.naturalWidth || 1)
    const scaleY = canvas.height / (detectionData.image_height || img.naturalHeight || 1)

    detectionData.detections.forEach((det) => {
      const [x1, y1, x2, y2] = det.bounding_box
      const boxX = x1 * scaleX
      const boxY = y1 * scaleY
      const boxW = (x2 - x1) * scaleX
      const boxH = (y2 - y1) * scaleY

      const color = CLASS_COLORS[det.class_name] || '#3b82f6'

      // Box border
      ctx.strokeStyle = color
      ctx.lineWidth = 2.5
      ctx.strokeRect(boxX, boxY, boxW, boxH)

      // Box fill
      ctx.fillStyle = `${color}25`
      ctx.fillRect(boxX, boxY, boxW, boxH)

      // Label background
      ctx.fillStyle = color
      const labelText = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`
      ctx.font = '11px JetBrains Mono, sans-serif'
      const textWidth = ctx.measureText(labelText).width + 8
      ctx.fillRect(boxX, Math.max(0, boxY - 18), textWidth, 18)

      // Label text
      ctx.fillStyle = '#ffffff'
      ctx.fillText(labelText, boxX + 4, Math.max(13, boxY - 4))
    })
  }, [detectionData, previewUrl])

  // Call POST /detect
  const handleDetect = async () => {
    if (!selectedFile) {
      setErrorMsg('Please select an invoice image or PDF file.')
      return
    }
    setLoading(true)
    setErrorMsg(null)

    const formData = new FormData()
    formData.append('file', selectedFile)
    if (documentId) formData.append('document_id', documentId)

    try {
      const res = await fetch(`${API_BASE}/detect`, {
        method: 'POST',
        body: formData,
      })
      const data = await res.json()
      if (!res.ok) {
        throw new Error(data.detail || 'Detection failed')
      }
      setDetectionData(data)
    } catch (err) {
      setErrorMsg(err.message)
    } finally {
      setLoading(false)
    }
  }

  // Call POST /process (Full end-to-end detection + reasoning)
  const handleProcess = async () => {
    if (!selectedFile) {
      setErrorMsg('Please select an invoice image or PDF file.')
      return
    }
    setLoading(true)
    setErrorMsg(null)

    const formData = new FormData()
    formData.append('file', selectedFile)
    if (documentId) formData.append('document_id', documentId)
    if (asOfDate) formData.append('as_of_date', asOfDate)

    try {
      const res = await fetch(`${API_BASE}/process`, {
        method: 'POST',
        body: formData,
      })
      const data = await res.json()
      if (!res.ok) {
        throw new Error(data.detail || 'Pipeline processing failed')
      }
      setProcessData(data)
      setDetectionData({
        document_id: data.document_id,
        image_width: data.evidence_request.fields?.[0]?.bbox ? 640 : 1000,
        image_height: 1000,
        detections: data.detections || [],
      })
    } catch (err) {
      setErrorMsg(err.message)
    } finally {
      setLoading(false)
    }
  }

  // Call POST /reason with JSON
  const handleDirectReason = async () => {
    setReasonLoading(true)
    try {
      const payload = JSON.parse(reasonJson)
      const res = await fetch(`${API_BASE}/reason`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await res.json()
      setReasonResult(data)
    } catch (err) {
      alert(`Invalid JSON or request failed: ${err.message}`)
    } finally {
      setReasonLoading(false)
    }
  }

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand-area">
          <div className="brand-logo">
            <ShieldCheck size={26} color="#ffffff" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <span className="brand-title">RecoverOps</span>
              <span className="brand-badge">RT-DETR-L + Guardrails</span>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
              Autonomous Invoice Detection & Deterministic Financial Recovery Reasoning
            </p>
          </div>
        </div>

        <div className="header-meta">
          <div className="status-pill">
            <span
              className={`status-dot ${
                backendHealth.status === 'online'
                  ? backendHealth.model_loaded
                    ? 'online'
                    : 'standby'
                  : 'offline'
              }`}
            />
            <span>
              {backendHealth.status === 'online'
                ? backendHealth.model_loaded
                  ? 'Backend: Online (Model Ready)'
                  : 'Backend: Online (Standby)'
                : 'Backend: Offline'}
            </span>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="tab-nav">
        <button
          className={`tab-btn ${activeTab === 'pipeline' ? 'active' : ''}`}
          onClick={() => setActiveTab('pipeline')}
        >
          <Layers size={16} /> End-to-End Pipeline
        </button>
        <button
          className={`tab-btn ${activeTab === 'reason' ? 'active' : ''}`}
          onClick={() => setActiveTab('reason')}
        >
          <Cpu size={16} /> Direct Reasoning API
        </button>
        <button
          className={`tab-btn ${activeTab === 'docs' ? 'active' : ''}`}
          onClick={() => setActiveTab('docs')}
        >
          <Info size={16} /> Architecture & Deploy
        </button>
      </nav>

      {/* TAB 1: Pipeline */}
      {activeTab === 'pipeline' && (
        <div className="main-grid">
          {/* Left Column: Input and Upload */}
          <div className="glass-card">
            <div className="card-header">
              <h2 className="card-title">
                <Upload size={18} color="#3b82f6" /> 1. Upload Invoice
              </h2>
              {selectedFile && (
                <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                  {(selectedFile.size / 1024).toFixed(1)} KB
                </span>
              )}
            </div>

            <div
              className="dropzone"
              onClick={() => document.getElementById('file-upload').click()}
              onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
              onDrop={(e) => {
                e.preventDefault();
                e.stopPropagation();
                if (e.dataTransfer.files?.[0]) handleFileSelect(e.dataTransfer.files[0])
              }}
            >
              <input
                id="file-upload"
                type="file"
                accept=".png,.jpg,.jpeg,.webp,.tiff,.bmp,.pdf"
                style={{ display: 'none' }}
                onChange={(e) => handleFileSelect(e.target.files?.[0])}
              />
              <FileText className="dropzone-icon" />
              <p className="dropzone-text">
                {selectedFile ? selectedFile.name : 'Click to browse or drop invoice image / PDF'}
              </p>
              <p className="dropzone-subtext">PNG, JPG, WEBP, PDF up to 25MB</p>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label className="form-label">Document ID</label>
                <input
                  type="text"
                  className="form-input"
                  placeholder="e.g. INV-2026-001"
                  value={documentId}
                  onChange={(e) => setDocumentId(e.target.value)}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Reference Date (As Of)</label>
                <input
                  type="date"
                  className="form-input"
                  value={asOfDate}
                  onChange={(e) => setAsOfDate(e.target.value)}
                />
              </div>
            </div>

            <div className="action-row">
              <button
                className="btn btn-secondary"
                disabled={!selectedFile || loading}
                onClick={handleDetect}
              >
                {loading ? <div className="spinner" /> : <Eye size={16} />}
                Detect Regions
              </button>
              <button
                className="btn btn-primary"
                disabled={!selectedFile || loading}
                onClick={handleProcess}
              >
                {loading ? <div className="spinner" /> : <Play size={16} />}
                Run Full Pipeline
              </button>
            </div>

            {errorMsg && (
              <div style={{ marginTop: '16px', padding: '12px', background: 'rgba(239, 68, 68, 0.15)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '8px', color: '#fca5a5', fontSize: '0.85rem', display: 'flex', gap: '8px', alignItems: 'center' }}>
                <AlertCircle size={18} />
                <span>{errorMsg}</span>
              </div>
            )}

            {/* Honest OCR Limitation Notice */}
            <div className="notice-box">
              <Info size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
              <div>
                <strong>OCR Integration Note:</strong> Part A RT-DETR-L extracts field <em>regions</em> (bounding boxes). Optical Character Recognition (OCR) text transcription is not yet connected; field values remain <code>null</code> to ensure strict financial safety.
              </div>
            </div>

            {/* Visual Preview */}
            {previewUrl && (
              <div className="preview-container">
                <img
                  ref={imageRef}
                  src={previewUrl}
                  alt="Invoice Preview"
                  className="preview-img"
                  onLoad={() => {
                    // Trigger canvas redraw
                    if (detectionData) setDetectionData({ ...detectionData })
                  }}
                />
                <canvas ref={canvasRef} className="bbox-canvas" />
              </div>
            )}
          </div>

          {/* Right Column: Results & Decisions */}
          <div className="glass-card">
            <div className="card-header">
              <h2 className="card-title">
                <ShieldCheck size={18} color="#10b981" /> 2. Detection & Reasoning Output
              </h2>
            </div>

            {!detectionData && !processData && (
              <div style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-dim)' }}>
                <Layers size={42} style={{ margin: '0 auto 12px', opacity: 0.4 }} />
                <p style={{ fontWeight: 600 }}>No inference executed yet</p>
                <p style={{ fontSize: '0.82rem' }}>Upload an invoice and click "Run Full Pipeline" to see live results.</p>
              </div>
            )}

            {/* Decision Banner */}
            {processData?.decision && (
              <div className={`decision-banner ${processData.decision.confidence_state}`}>
                <div className="decision-header-row">
                  <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    <span className={`state-badge ${processData.decision.confidence_state}`}>
                      {processData.decision.confidence_state}
                    </span>
                    <span className="intent-badge">
                      {processData.decision.intent}
                    </span>
                  </div>
                  {processData.decision.requires_human_review && (
                    <span style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#f59e0b', fontSize: '0.78rem', fontWeight: 600 }}>
                      <AlertTriangle size={14} /> Human Review Required
                    </span>
                  )}
                </div>

                <div className="decision-action">
                  {processData.decision.recommended_action}
                </div>
                <div className="decision-reason">
                  {processData.decision.reason}
                </div>
              </div>
            )}

            {/* Detected Field Regions */}
            {detectionData && (
              <div className="result-block">
                <h3 style={{ fontSize: '0.95rem', marginBottom: '10px', color: 'var(--text-muted)' }}>
                  Detected Field Regions ({detectionData.detections?.length || 0})
                </h3>
                <div className="detection-list">
                  {detectionData.detections?.length === 0 ? (
                    <p style={{ fontSize: '0.82rem', color: 'var(--text-dim)' }}>No invoice field regions detected.</p>
                  ) : (
                    detectionData.detections?.map((det, i) => (
                      <div key={i} className="detection-item">
                        <div className="detection-tag">
                          <span className={`tag-dot ${det.class_name}`} />
                          <span>{det.class_name}</span>
                        </div>
                        <span className="detection-conf">
                          conf: {(det.confidence * 100).toFixed(1)}% | [
                          {det.bounding_box.map((c) => Math.round(c)).join(', ')}]
                        </span>
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}

            {/* Guardrail Checks */}
            {processData?.decision?.guardrail_checks && (
              <div className="result-block">
                <h3 style={{ fontSize: '0.95rem', marginBottom: '8px', color: 'var(--text-muted)' }}>
                  Financial Guardrail Audit (G1–G6)
                </h3>
                <div className="checks-grid">
                  {processData.decision.guardrail_checks.map((chk, i) => (
                    <div key={i} className={`check-chip ${chk.passed ? 'passed' : 'failed'}`}>
                      <span>{chk.check_id}</span>
                      <span>{chk.passed ? 'PASSED' : 'FLAGGED'}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Raw JSON Trace */}
            {(processData || detectionData) && (
              <div className="result-block" style={{ marginTop: '24px' }}>
                <h3 style={{ fontSize: '0.85rem', marginBottom: '8px', color: 'var(--text-muted)' }}>
                  Raw Decision JSON (Auditable)
                </h3>
                <div className="json-viewer">
                  {JSON.stringify(processData?.decision || detectionData, null, 2)}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 2: Direct Reasoning API */}
      {activeTab === 'reason' && (
        <div className="main-grid">
          <div className="glass-card">
            <div className="card-header">
              <h2 className="card-title">
                <Cpu size={18} color="#8b5cf6" /> POST /reason Payload
              </h2>
              <div style={{ display: 'flex', gap: '6px' }}>
                {Object.keys(SAMPLE_REASONING_PRESETS).map((k) => (
                  <button
                    key={k}
                    className="btn btn-secondary"
                    style={{ fontSize: '0.75rem', padding: '4px 8px' }}
                    onClick={() => setReasonJson(JSON.stringify(SAMPLE_REASONING_PRESETS[k], null, 2))}
                  >
                    {k}
                  </button>
                ))}
              </div>
            </div>

            <textarea
              className="form-input"
              style={{ width: '100%', height: '340px', fontFamily: 'var(--font-mono)', fontSize: '0.82rem' }}
              value={reasonJson}
              onChange={(e) => setReasonJson(e.target.value)}
            />

            <button
              className="btn btn-primary"
              style={{ marginTop: '16px', width: '100%' }}
              disabled={reasonLoading}
              onClick={handleDirectReason}
            >
              {reasonLoading ? <div className="spinner" /> : <Play size={16} />}
              Evaluate Decision (POST /reason)
            </button>
          </div>

          <div className="glass-card">
            <div className="card-header">
              <h2 className="card-title">
                <CheckCircle2 size={18} color="#10b981" /> Decision Response
              </h2>
            </div>
            {reasonResult ? (
              <div>
                <div className={`decision-banner ${reasonResult.confidence_state}`}>
                  <div className="decision-header-row">
                    <span className={`state-badge ${reasonResult.confidence_state}`}>
                      {reasonResult.confidence_state}
                    </span>
                    <span className="intent-badge">{reasonResult.intent}</span>
                  </div>
                  <div className="decision-action">{reasonResult.recommended_action}</div>
                  <div className="decision-reason">{reasonResult.reason}</div>
                </div>
                <div className="json-viewer">
                  {JSON.stringify(reasonResult, null, 2)}
                </div>
              </div>
            ) : (
              <p style={{ color: 'var(--text-dim)', fontSize: '0.88rem' }}>Click "Evaluate Decision" to inspect the JSON response.</p>
            )}
          </div>
        </div>
      )}

      {/* TAB 3: Documentation & Deployment */}
      {activeTab === 'docs' && (
        <div className="glass-card">
          <div className="card-header">
            <h2 className="card-title">
              <Server size={18} color="#0ea5e9" /> Production Architecture & Deployment Guide
            </h2>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px', marginTop: '14px' }}>
            <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '18px', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
              <h3 style={{ color: '#93c5fd', fontSize: '1rem', marginBottom: '8px' }}>1. FastAPI Backend (Render)</h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
                Containerized with Docker on Python 3.12-slim. Exposes <code>/health</code>, <code>/reason</code>, <code>/detect</code>, and <code>/process</code>.
              </p>
              <pre style={{ background: '#090d16', padding: '10px', borderRadius: '6px', fontSize: '0.75rem', marginTop: '10px' }}>
                uvicorn src.app:app --host 0.0.0.0 --port $PORT
              </pre>
            </div>

            <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '18px', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
              <h3 style={{ color: '#a7f3d0', fontSize: '1rem', marginBottom: '8px' }}>2. Frontend (Vercel)</h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
                React + Vite client connecting over HTTPS to the Render backend via <code>VITE_API_URL</code>.
              </p>
              <pre style={{ background: '#090d16', padding: '10px', borderRadius: '6px', fontSize: '0.75rem', marginTop: '10px' }}>
                VITE_API_URL=https://&lt;render-app&gt;.onrender.com
              </pre>
            </div>

            <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '18px', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
              <h3 style={{ color: '#fde68a', fontSize: '1rem', marginBottom: '8px' }}>3. Model Strategy</h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
                Checkpoint resolved via <code>MODEL_PATH</code> (default <code>models/best.pt</code>). Fast failover with 503 if weights are not yet loaded.
              </p>
              <pre style={{ background: '#090d16', padding: '10px', borderRadius: '6px', fontSize: '0.75rem', marginTop: '10px' }}>
                MODEL_PATH=/app/models/best.pt
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
