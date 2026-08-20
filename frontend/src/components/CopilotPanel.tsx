import { useState } from 'react'
import { apiErrorMessage, copilotAsk, copilotReport } from '../api/client'
import type { AnalysisResponse, CopilotResponse } from '../api/types'
import { Spinner } from './XrayUpload'

interface ChatTurn {
  role: 'user' | 'assistant'
  content: string
}

interface Props {
  analysis: AnalysisResponse
  onCopilotResult: (result: CopilotResponse) => void
}

const SUGGESTED_QUESTIONS = [
  'Why did the model flag this?',
  'What does the confidence score mean?',
  'Summarize the AI findings.',
]

export default function CopilotPanel({ analysis, onCopilotResult }: Props) {
  const [history, setHistory] = useState<ChatTurn[]>([])
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [latestResult, setLatestResult] = useState<CopilotResponse | null>(null)

  async function generatePreliminaryReport() {
    setLoading(true)
    setError(null)
    try {
      const result = await copilotReport(
        analysis.vision.findings,
        analysis.vision.localization,
        analysis.study.modality,
      )
      setLatestResult(result)
      onCopilotResult(result)
      setHistory((h) => [...h, { role: 'assistant', content: result.summary }])
    } catch (e) {
      setError(apiErrorMessage(e))
    } finally {
      setLoading(false)
    }
  }

  async function ask(q: string) {
    if (!q.trim()) return
    setLoading(true)
    setError(null)
    const nextHistory: ChatTurn[] = [...history, { role: 'user', content: q }]
    setHistory(nextHistory)
    setQuestion('')
    try {
      const result = await copilotAsk(
        q,
        analysis.vision.findings,
        analysis.vision.localization,
        analysis.study.modality,
        history.map((t) => ({ role: t.role, content: t.content })),
      )
      setLatestResult(result)
      onCopilotResult(result)
      setHistory((h) => [...h, { role: 'assistant', content: result.summary }])
    } catch (e) {
      setError(apiErrorMessage(e))
      setHistory(history) // roll back the optimistic user turn's trailing state on failure
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card flex flex-col h-[520px]">
      <div className="flex items-center justify-between mb-3">
        <div className="text-xs font-medium text-ink-faint uppercase tracking-wide">AI Copilot</div>
        {latestResult && (
          <span className="text-[10px] font-mono-tabular text-ink-faint">
            {latestResult.provider}/{latestResult.model}
          </span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto space-y-3 pr-1">
        {history.length === 0 && !loading && (
          <div>
            <p className="text-sm text-ink-muted mb-3">
              Ask about this analysis, or generate a grounded preliminary report.
            </p>
            <button className="btn-secondary w-full justify-center" onClick={generatePreliminaryReport}>
              Generate Preliminary Report
            </button>
            <div className="mt-4 space-y-1.5">
              {SUGGESTED_QUESTIONS.map((q) => (
                <button
                  key={q}
                  className="w-full text-left text-xs text-ink-muted hover:text-accent px-2.5 py-1.5
                    rounded-md hover:bg-surface-2 transition-colors"
                  onClick={() => ask(q)}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

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

      {latestResult && latestResult.limitations.length > 0 && (
        <div className="mt-2 mb-1 text-[11px] text-ink-faint leading-snug border-t border-border pt-2">
          <span className="text-warn font-medium">Limitations:</span> {latestResult.limitations.join(' ')}
        </div>
      )}

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
          placeholder="Ask about this study…"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={loading}
        />
        <button className="btn-primary !py-2" type="submit" disabled={loading || !question.trim()}>
          Ask
        </button>
      </form>
    </div>
  )
}
