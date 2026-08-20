import { useEffect, useMemo, useState } from 'react'
import { apiErrorMessage, copilotAsk, getReport, listReports } from '../api/client'
import type { ReportDetail, ReportSummary } from '../api/types'
import { Spinner } from '../components/XrayUpload'

// A standalone "AI Copilot" page, matching the mockup's dedicated chat
// nav item -- but grounded in a real past study's real findings, never a
// freeform chat with no vision context. The project's whole safety
// design (src/safety/groundedness.py) exists specifically so the LLM
// never answers about findings it wasn't given; a context-free chat page
// would defeat that by construction, so this page always requires
// picking a real analyzed study first.

interface ChatTurn {
  role: 'user' | 'assistant'
  content: string
}

export default function CopilotChat() {
  const [reports, setReports] = useState<ReportSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selectedReportId, setSelectedReportId] = useState('')
  const [detail, setDetail] = useState<ReportDetail | null>(null)
  const [candidateIndex, setCandidateIndex] = useState(0)

  useEffect(() => {
    listReports(100)
      .then(setReports)
      .catch((e) => setError(apiErrorMessage(e)))
  }, [])

  useEffect(() => {
    if (!selectedReportId) {
      setDetail(null)
      return
    }
    setDetail(null)
    setCandidateIndex(0)
    getReport(selectedReportId)
      .then(setDetail)
      .catch((e) => setError(apiErrorMessage(e)))
  }, [selectedReportId])

  return (
    <div className="p-8 max-w-5xl">
      <h1 className="text-xl font-semibold tracking-tight mb-1">AI Copilot</h1>
      <p className="text-sm text-ink-muted mb-6">
        Ask questions grounded in a real, previously analyzed study — the Copilot only ever sees the actual
        vision findings for the study you pick below, never a freeform image-less chat.
      </p>

      {error && (
        <div className="rounded-lg border border-danger/40 bg-danger-soft px-4 py-2.5 text-sm text-danger mb-6">
          {error}
        </div>
      )}

      {reports === null && !error && <div className="text-sm text-ink-faint">Loading…</div>}

      {reports !== null && reports.length === 0 && (
        <div className="card text-sm text-ink-muted">
          No analyzed studies yet — generate a report from a new study first, then come back here to ask
          follow-up questions about it.
        </div>
      )}

      {reports !== null && reports.length > 0 && (
        <div className="grid grid-cols-[280px_1fr] gap-5">
          <div className="card !p-0 overflow-hidden self-start">
            <div className="px-4 py-3 text-xs font-medium text-ink-faint uppercase tracking-wide border-b border-border">
              Select a Study
            </div>
            <ul className="max-h-[520px] overflow-y-auto">
              {reports.map((r) => (
                <li key={r.report_id} className="border-b border-border last:border-0">
                  <button
                    type="button"
                    onClick={() => setSelectedReportId(r.report_id)}
                    className={`w-full text-left px-4 py-3 transition-colors ${
                      selectedReportId === r.report_id ? 'bg-accent-soft' : 'hover:bg-surface-2'
                    }`}
                  >
                    <div className="text-sm text-ink truncate">{r.patient_name}</div>
                    <div className="text-[11px] text-ink-faint font-mono-tabular mt-0.5">
                      {r.modality.toUpperCase()} · {r.primary_finding ?? '—'}
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          </div>

          {detail ? (
            <CandidateChat
              key={`${detail.report_id}-${candidateIndex}`}
              detail={detail}
              candidateIndex={candidateIndex}
              onCandidateChange={setCandidateIndex}
            />
          ) : selectedReportId ? (
            <div className="card flex items-center justify-center text-sm text-ink-faint h-[300px]">
              <Spinner /> <span className="ml-2">Loading study…</span>
            </div>
          ) : (
            <div className="card flex items-center justify-center text-sm text-ink-faint h-[300px] text-center px-8">
              Pick a study on the left to start a grounded conversation about its findings.
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function CandidateChat({
  detail,
  candidateIndex,
  onCandidateChange,
}: {
  detail: ReportDetail
  candidateIndex: number
  onCandidateChange: (i: number) => void
}) {
  const candidate = detail.candidates[candidateIndex]
  const modality = detail.modality as 'xray' | 'ct'

  const [history, setHistory] = useState<ChatTurn[]>(
    candidate ? [{ role: 'assistant', content: candidate.copilot_summary }] : [],
  )
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const findings = useMemo(() => candidate?.vision_findings ?? [], [candidate])

  async function ask(q: string) {
    if (!q.trim() || !candidate) return
    setLoading(true)
    setError(null)
    const nextHistory: ChatTurn[] = [...history, { role: 'user', content: q }]
    setHistory(nextHistory)
    setQuestion('')
    try {
      const result = await copilotAsk(
        q,
        findings,
        candidate.localization,
        modality,
        history.map((t) => ({ role: t.role, content: t.content })),
      )
      setHistory((h) => [...h, { role: 'assistant', content: result.summary }])
    } catch (e) {
      setError(apiErrorMessage(e))
      setHistory(history)
    } finally {
      setLoading(false)
    }
  }

  if (!candidate) return null

  return (
    <div className="flex flex-col gap-3">
      <div className="card">
        <div className="flex items-center justify-between mb-3">
          <div className="text-xs font-medium text-ink-faint uppercase tracking-wide">
            {detail.patient_name} — {detail.study_id}
          </div>
          {detail.candidates.length > 1 && (
            <div className="flex items-center gap-1">
              {detail.candidates.map((_, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => onCandidateChange(i)}
                  className={`text-[11px] px-2 py-1 rounded-md font-mono-tabular ${
                    i === candidateIndex ? 'bg-accent-soft text-accent' : 'text-ink-faint hover:bg-surface-2'
                  }`}
                >
                  Loc {i + 1}
                </button>
              ))}
            </div>
          )}
        </div>
        <div className="flex flex-wrap gap-2">
          {findings.map((f, i) => (
            <span key={i} className="text-xs font-mono-tabular px-2 py-1 rounded-md bg-surface-2 text-ink-muted">
              {f.label} · {Math.min(f.probability * 100, 99.99).toFixed(2)}%
            </span>
          ))}
          {candidate.localization && (
            <span className="text-xs font-mono-tabular px-2 py-1 rounded-md bg-surface-2 text-ink-muted">
              {candidate.localization}
            </span>
          )}
        </div>
      </div>

      <div className="card flex flex-col h-[420px]">
        <div className="flex-1 overflow-y-auto space-y-3 pr-1">
          {history.map((turn, i) => (
            <div
              key={i}
              className={`text-sm rounded-lg px-3 py-2 max-w-[92%] ${
                turn.role === 'user'
                  ? 'ml-auto bg-accent-soft text-accent border border-accent/30'
                  : 'bg-surface-2 text-ink'
              }`}
            >
              {turn.content}
            </div>
          ))}
          {loading && (
            <div className="flex items-center gap-2 text-xs text-ink-faint">
              <Spinner /> thinking…
            </div>
          )}
        </div>

        {error && <div className="mt-2 text-xs text-danger">{error}</div>}

        <form
          className="mt-3 flex gap-2"
          onSubmit={(e) => {
            e.preventDefault()
            ask(question)
          }}
        >
          <input
            className="input"
            placeholder="Ask a follow-up about this study…"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={loading}
          />
          <button className="btn-primary !py-2" type="submit" disabled={loading || !question.trim()}>
            Ask
          </button>
        </form>
      </div>
    </div>
  )
}
