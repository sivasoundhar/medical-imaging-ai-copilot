import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { apiErrorMessage, deleteReport, listReports } from '../api/client'
import type { ReportSummary } from '../api/types'
import ConfirmDialog from '../components/ConfirmDialog'
import { TrashIcon } from '../components/icons'

export default function ReportsHistory() {
  const [reports, setReports] = useState<ReportSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [pendingDelete, setPendingDelete] = useState<ReportSummary | null>(null)
  const [deleting, setDeleting] = useState(false)

  useEffect(() => {
    listReports()
      .then(setReports)
      .catch((e) => setError(apiErrorMessage(e)))
  }, [])

  async function confirmDelete() {
    if (!pendingDelete) return
    setDeleting(true)
    try {
      await deleteReport(pendingDelete.report_id)
      setReports((rs) => (rs ? rs.filter((r) => r.report_id !== pendingDelete.report_id) : rs))
      setPendingDelete(null)
    } catch (e) {
      setError(apiErrorMessage(e))
    } finally {
      setDeleting(false)
    }
  }

  return (
    <div className="p-8 max-w-4xl">
      <h1 className="text-xl font-semibold tracking-tight mb-1">Reports History</h1>
      <p className="text-sm text-ink-muted mb-6">Every report generated in this session, backed by real storage.</p>

      {error && (
        <div className="rounded-lg border border-danger/40 bg-danger-soft px-4 py-2.5 text-sm text-danger mb-4">
          {error}
        </div>
      )}

      {reports === null && !error && <div className="text-sm text-ink-faint">Loading…</div>}

      {reports !== null && reports.length === 0 && (
        <div className="card text-sm text-ink-muted">
          No reports yet.{' '}
          <Link to="/new-study" className="text-accent hover:underline">
            Start a new study
          </Link>{' '}
          to generate one.
        </div>
      )}

      {reports !== null && reports.length > 0 && (
        <div className="card !p-0 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-ink-faint uppercase tracking-wide border-b border-border">
                <th className="px-4 py-3 font-medium">Patient</th>
                <th className="px-4 py-3 font-medium">Modality</th>
                <th className="px-4 py-3 font-medium">Finding</th>
                <th className="px-4 py-3 font-medium">Locations</th>
                <th className="px-4 py-3 font-medium">Study Date</th>
                <th className="px-4 py-3 font-medium">Generated</th>
                <th className="px-4 py-3 font-medium" />
              </tr>
            </thead>
            <tbody>
              {reports.map((r) => (
                <tr key={r.report_id} className="border-b border-border last:border-0 hover:bg-surface-2">
                  <td className="px-4 py-3">
                    <Link to={`/reports/${r.report_id}`} className="block">
                      {r.patient_name}
                    </Link>
                  </td>
                  <td className="px-4 py-3 uppercase text-ink-muted font-mono-tabular text-xs">{r.modality}</td>
                  <td className="px-4 py-3 text-ink-muted">{r.primary_finding ?? '—'}</td>
                  <td className="px-4 py-3 text-ink-muted font-mono-tabular text-xs">
                    {r.location_count > 1 ? `${r.location_count} locations` : '1'}
                  </td>
                  <td className="px-4 py-3 text-ink-muted font-mono-tabular text-xs">{r.study_date}</td>
                  <td className="px-4 py-3 text-ink-faint font-mono-tabular text-xs">
                    {new Date(r.created_at).toLocaleString()}
                  </td>
                  <td className="px-4 py-3">
                    <button
                      type="button"
                      title="Delete report"
                      onClick={() => setPendingDelete(r)}
                      className="w-7 h-7 rounded-md flex items-center justify-center text-ink-faint hover:bg-danger-soft hover:text-danger transition-colors"
                    >
                      <TrashIcon className="w-4 h-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <ConfirmDialog
        open={pendingDelete !== null}
        title="Delete this report?"
        description={
          pendingDelete
            ? `This permanently deletes the report for ${pendingDelete.patient_name} (${pendingDelete.study_id}) and its PDF file. This cannot be undone.`
            : ''
        }
        confirmLabel="Delete Report"
        busy={deleting}
        onConfirm={confirmDelete}
        onCancel={() => setPendingDelete(null)}
      />
    </div>
  )
}
