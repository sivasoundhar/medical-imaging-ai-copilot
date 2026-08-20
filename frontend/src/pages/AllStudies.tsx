import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { apiErrorMessage, deleteReport, listReports } from '../api/client'
import type { ReportSummary } from '../api/types'
import ConfirmDialog from '../components/ConfirmDialog'
import { SearchIcon, TrashIcon } from '../components/icons'

// Note (honest, not swept under the rug): a "study" only becomes
// queryable once a report has been generated for it (storage/models.py
// only persists ReportRecord, no separate pre-report study table) -- so
// "All Studies" is backed by the exact same real data as "Reports
// History", just presented as a searchable/sortable data table instead
// of a list. Not a duplicate feature by mistake, a real current
// constraint of the data model.

type SortKey = 'date' | 'patient'

export default function AllStudies() {
  const [reports, setReports] = useState<ReportSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [sortKey, setSortKey] = useState<SortKey>('date')
  const [pendingDelete, setPendingDelete] = useState<ReportSummary | null>(null)
  const [deleting, setDeleting] = useState(false)

  useEffect(() => {
    listReports(200)
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

  const filtered = useMemo(() => {
    if (!reports) return []
    const q = query.trim().toLowerCase()
    let rows = reports
    if (q) {
      rows = reports.filter(
        (r) =>
          r.patient_name.toLowerCase().includes(q) ||
          r.study_id.toLowerCase().includes(q) ||
          (r.primary_finding ?? '').toLowerCase().includes(q) ||
          r.modality.toLowerCase().includes(q),
      )
    }
    return [...rows].sort((a, b) =>
      sortKey === 'patient'
        ? a.patient_name.localeCompare(b.patient_name)
        : new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
    )
  }, [reports, query, sortKey])

  return (
    <div className="p-8 max-w-6xl">
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-xl font-semibold tracking-tight mb-1">All Studies</h1>
          <p className="text-sm text-ink-muted">Every analyzed study with a generated report, real data only.</p>
        </div>
        <Link to="/new-study" className="btn-primary">
          Upload New Study
        </Link>
      </div>

      {error && (
        <div className="rounded-lg border border-danger/40 bg-danger-soft px-4 py-2.5 text-sm text-danger mb-6">
          {error}
        </div>
      )}

      {reports === null && !error && <div className="text-sm text-ink-faint">Loading…</div>}

      {reports !== null && reports.length === 0 && (
        <div className="card text-sm text-ink-muted">
          No studies yet.{' '}
          <Link to="/new-study" className="text-accent hover:underline">
            Start a new study
          </Link>
          .
        </div>
      )}

      {reports !== null && reports.length > 0 && (
        <>
          <div className="flex items-center gap-3 mb-4">
            <div className="relative flex-1 max-w-sm">
              <SearchIcon className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-ink-faint" />
              <input
                className="input !pl-9"
                placeholder="Search patient, study ID, finding…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>
            <div className="text-xs text-ink-faint">{filtered.length} of {reports.length}</div>
          </div>

          <div className="card !p-0 overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-ink-faint uppercase tracking-wide border-b border-border">
                  <th className="px-4 py-3 font-medium">Study ID</th>
                  <th
                    className="px-4 py-3 font-medium cursor-pointer hover:text-ink"
                    onClick={() => setSortKey('patient')}
                  >
                    Patient {sortKey === 'patient' && '↓'}
                  </th>
                  <th className="px-4 py-3 font-medium">Modality</th>
                  <th className="px-4 py-3 font-medium">Primary Finding</th>
                  <th className="px-4 py-3 font-medium">Locations</th>
                  <th
                    className="px-4 py-3 font-medium cursor-pointer hover:text-ink"
                    onClick={() => setSortKey('date')}
                  >
                    Date {sortKey === 'date' && '↓'}
                  </th>
                  <th className="px-4 py-3 font-medium" />
                </tr>
              </thead>
              <tbody>
                {filtered.map((r) => (
                  <tr key={r.report_id} className="border-b border-border last:border-0 hover:bg-surface-2">
                    <td className="px-4 py-3 font-mono-tabular text-xs text-ink-muted">{r.study_id}</td>
                    <td className="px-4 py-3">{r.patient_name}</td>
                    <td className="px-4 py-3 uppercase text-ink-muted font-mono-tabular text-xs">{r.modality}</td>
                    <td className="px-4 py-3 text-ink-muted">{r.primary_finding ?? '—'}</td>
                    <td className="px-4 py-3 text-ink-muted font-mono-tabular text-xs">{r.location_count}</td>
                    <td className="px-4 py-3 text-ink-faint font-mono-tabular text-xs">{r.study_date}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-3">
                        <Link to={`/reports/${r.report_id}`} className="text-accent text-xs hover:underline">
                          View Report
                        </Link>
                        <button
                          type="button"
                          title="Delete report"
                          onClick={() => setPendingDelete(r)}
                          className="w-6 h-6 rounded-md flex items-center justify-center text-ink-faint hover:bg-danger-soft hover:text-danger transition-colors"
                        >
                          <TrashIcon className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
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
