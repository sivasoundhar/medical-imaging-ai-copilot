import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { apiErrorMessage, deleteReport, getReport, reportPdfUrl } from '../api/client'
import type { ReportDetail } from '../api/types'
import ConfirmDialog from '../components/ConfirmDialog'
import { TrashIcon } from '../components/icons'

export default function ReportView() {
  const { reportId } = useParams<{ reportId: string }>()
  const navigate = useNavigate()
  const [report, setReport] = useState<ReportDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [confirmingDelete, setConfirmingDelete] = useState(false)
  const [deleting, setDeleting] = useState(false)

  useEffect(() => {
    if (!reportId) return
    getReport(reportId)
      .then(setReport)
      .catch((e) => setError(apiErrorMessage(e)))
  }, [reportId])

  async function confirmDelete() {
    if (!reportId) return
    setDeleting(true)
    try {
      await deleteReport(reportId)
      navigate('/reports')
    } catch (e) {
      setError(apiErrorMessage(e))
      setDeleting(false)
      setConfirmingDelete(false)
    }
  }

  if (error) {
    return (
      <div className="p-8 max-w-2xl">
        <div className="rounded-lg border border-danger/40 bg-danger-soft px-4 py-2.5 text-sm text-danger">
          {error}
        </div>
        <Link to="/reports" className="mt-4 inline-block text-sm text-accent hover:underline">
          ← Back to Reports History
        </Link>
      </div>
    )
  }

  if (!report) return <div className="p-8 text-sm text-ink-faint">Loading…</div>

  return (
    <div className="p-8 max-w-3xl">
      <Link to="/reports" className="text-xs text-ink-faint hover:text-accent">
        ← Reports History
      </Link>

      <div className="mt-3 flex items-start justify-between">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">{report.patient_name}</h1>
          <p className="text-sm text-ink-muted font-mono-tabular">
            {report.study_id} · {report.modality.toUpperCase()} · {report.study_date}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <a href={reportPdfUrl(report.report_id)} target="_blank" rel="noreferrer" className="btn-primary">
            Download PDF
          </a>
          <button
            type="button"
            title="Delete report"
            onClick={() => setConfirmingDelete(true)}
            className="w-9 h-9 rounded-lg border border-border flex items-center justify-center text-ink-faint hover:bg-danger-soft hover:text-danger hover:border-danger/40 transition-colors"
          >
            <TrashIcon className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="mt-6 card">
        <SectionLabel>Vision Model</SectionLabel>
        <p className="mt-1 text-sm text-ink-muted">{report.vision_model_version}</p>
      </div>

      {/* One section per analyzed location. X-ray reports always have
          exactly one; CT reports can have several (Day 11.1) -- e.g.
          multiple candidate nodules checked on the same scan. */}
      {report.candidates.map((c, i) => (
        <div key={i} className="mt-5">
          {report.candidates.length > 1 && (
            <div className="mb-2 text-xs font-semibold text-accent uppercase tracking-wide">
              Location {i + 1} of {report.candidates.length}
            </div>
          )}
          <div className="grid grid-cols-2 gap-5">
            <div className="card">
              <SectionLabel>AI Vision Findings</SectionLabel>
              <div className="space-y-2 mt-2">
                {c.vision_findings.map((f, j) => (
                  <div key={j} className="flex justify-between text-sm">
                    <span>{f.label}</span>
                    <span className="font-mono-tabular text-ink-muted">
                      {Math.min(f.probability * 100, 99.99).toFixed(2)}%
                    </span>
                  </div>
                ))}
              </div>
              {c.localization && (
                <div className="mt-3 pt-3 border-t border-border text-xs font-mono-tabular text-ink-muted">
                  {c.localization}
                </div>
              )}
            </div>

            <div className="card">
              <SectionLabel>Model &amp; Provider</SectionLabel>
              <dl className="mt-2 space-y-1.5 text-sm">
                <Row label="LLM" value={`${c.llm_provider} / ${c.llm_model}`} />
                <Row label="Professional review" value={c.requires_professional_review ? 'Required' : '—'} />
              </dl>
            </div>
          </div>

          <div className="card mt-3">
            <SectionLabel>AI-Generated Preliminary Impression</SectionLabel>
            <p className="mt-2 text-sm leading-relaxed text-ink">{c.copilot_summary}</p>
            {c.copilot_limitations.length > 0 && (
              <div className="mt-4 pt-3 border-t border-border">
                <SectionLabel>Limitations</SectionLabel>
                <ul className="mt-2 space-y-1 text-sm text-ink-muted list-disc list-inside">
                  {c.copilot_limitations.map((l, j) => (
                    <li key={j}>{l}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      ))}

      <div className="mt-4 rounded-lg border border-warn-border bg-warn-soft px-4 py-3 text-xs text-ink-muted leading-relaxed">
        <span className="text-warn font-semibold">AI-generated output.</span> This is a research/portfolio
        prototype, not a clinical device. Model probability is not clinical certainty. Requires
        professional review.
      </div>

      <ConfirmDialog
        open={confirmingDelete}
        title="Delete this report?"
        description={`This permanently deletes the report for ${report.patient_name} (${report.study_id}) and its PDF file. This cannot be undone.`}
        confirmLabel="Delete Report"
        busy={deleting}
        onConfirm={confirmDelete}
        onCancel={() => setConfirmingDelete(false)}
      />
    </div>
  )
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return <div className="text-xs font-medium text-ink-faint uppercase tracking-wide">{children}</div>
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-ink-faint">{label}</span>
      <span className="text-ink-muted">{value}</span>
    </div>
  )
}
