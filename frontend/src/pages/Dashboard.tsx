import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { apiErrorMessage, listReports } from '../api/client'
import type { ReportSummary } from '../api/types'
import StatTile from '../components/StatTile'
import BarChart, { type BarDatum } from '../components/BarChart'

// The only two canonical modalities/label sets the real models produce
// (see docs/MODEL_CARD.md) -- used to keep the finding-distribution chart
// to real, fixed categories rather than an unbounded list of free text.
const FINDING_COLORS: Record<string, string> = {
  NORMAL: 'var(--color-chart-3)',
  PNEUMONIA: 'var(--color-chart-2)',
  NODULE: 'var(--color-chart-4)',
  NON_NODULE: 'var(--color-chart-3)',
}

export default function Dashboard() {
  const [reports, setReports] = useState<ReportSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listReports(100)
      .then(setReports)
      .catch((e) => setError(apiErrorMessage(e)))
  }, [])

  const stats = useMemo(() => {
    if (!reports) return null
    const xray = reports.filter((r) => r.modality === 'xray').length
    const ct = reports.filter((r) => r.modality === 'ct').length
    const locations = reports.reduce((sum, r) => sum + r.location_count, 0)
    const findingCounts = new Map<string, number>()
    for (const r of reports) {
      const key = (r.primary_finding ?? 'Unknown').toUpperCase()
      findingCounts.set(key, (findingCounts.get(key) ?? 0) + 1)
    }
    const findingData: BarDatum[] = [...findingCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6)
      .map(([label, value]) => ({
        label,
        value,
        colorVar: FINDING_COLORS[label] ?? 'var(--color-chart-1)',
      }))
    return { xray, ct, locations, findingData, recent: reports.slice(0, 6) }
  }, [reports])

  return (
    <div className="p-8 max-w-6xl">
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-xl font-semibold tracking-tight mb-1">Dashboard</h1>
          <p className="text-sm text-ink-muted">
            Real aggregate activity across every report generated in this deployment.
          </p>
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
          No studies analyzed yet.{' '}
          <Link to="/new-study" className="text-accent hover:underline">
            Analyze your first study
          </Link>{' '}
          to populate this dashboard with real data.
        </div>
      )}

      {stats && reports && reports.length > 0 && (
        <>
          <div className="grid grid-cols-4 gap-4 mb-5">
            <StatTile label="Total studies" value={reports.length} accent="accent" />
            <StatTile label="X-ray studies" value={stats.xray} accent="positive" />
            <StatTile label="CT studies" value={stats.ct} accent="warn" />
            <StatTile
              label="Candidate locations analyzed"
              value={stats.locations}
              hint="Includes multi-nodule CT reports"
              accent="accent"
            />
          </div>

          <div className="grid grid-cols-[1.2fr_1fr] gap-5">
            <div className="card">
              <div className="text-xs font-medium text-ink-faint uppercase tracking-wide mb-4">
                Primary Finding Distribution
              </div>
              <BarChart data={stats.findingData} />
            </div>

            <div className="card !p-0 overflow-hidden flex flex-col">
              <div className="px-5 py-4 text-xs font-medium text-ink-faint uppercase tracking-wide border-b border-border">
                Recent Studies
              </div>
              <ul className="flex-1">
                {stats.recent.map((r) => (
                  <li key={r.report_id} className="border-b border-border last:border-0">
                    <Link to={`/reports/${r.report_id}`} className="flex items-center justify-between px-5 py-3 hover:bg-surface-2 transition-colors">
                      <div className="min-w-0">
                        <div className="text-sm text-ink truncate">{r.patient_name}</div>
                        <div className="text-[11px] text-ink-faint font-mono-tabular mt-0.5">
                          {r.modality.toUpperCase()} · {r.primary_finding ?? '—'}
                        </div>
                      </div>
                      <div className="text-[10.5px] text-ink-faint font-mono-tabular shrink-0 ml-3">
                        {new Date(r.created_at).toLocaleDateString()}
                      </div>
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
