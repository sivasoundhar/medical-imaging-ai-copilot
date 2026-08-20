import { useEffect, useMemo, useState } from 'react'
import { apiErrorMessage, getReport, listReports } from '../api/client'
import type { ReportDetail, ReportSummary } from '../api/types'
import StatTile from '../components/StatTile'
import BarChart, { type BarDatum } from '../components/BarChart'

const DEEP_DIVE_LIMIT = 20

const FINDING_COLORS: Record<string, string> = {
  NORMAL: 'var(--color-chart-3)',
  PNEUMONIA: 'var(--color-chart-2)',
  NODULE: 'var(--color-chart-4)',
  NON_NODULE: 'var(--color-chart-3)',
}

const PROVIDER_COLORS: Record<string, string> = {
  groq: 'var(--color-chart-1)',
  ollama: 'var(--color-chart-2)',
  claude: 'var(--color-chart-3)',
  mock: 'var(--color-chart-4)',
}

function last14Days(): string[] {
  const days: string[] = []
  for (let i = 13; i >= 0; i--) {
    const d = new Date()
    d.setDate(d.getDate() - i)
    days.push(d.toISOString().slice(0, 10))
  }
  return days
}

export default function Analytics() {
  const [reports, setReports] = useState<ReportSummary[] | null>(null)
  const [details, setDetails] = useState<ReportDetail[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listReports(100)
      .then((rs) => {
        setReports(rs)
        // "Deeper insights" below is deliberately capped -- fetching full
        // per-candidate detail for every report doesn't scale, and this
        // is honestly labeled as based on a recent subset, not "all
        // reports," in the UI.
        const subset = rs.slice(0, DEEP_DIVE_LIMIT)
        return Promise.all(subset.map((r) => getReport(r.report_id)))
      })
      .then(setDetails)
      .catch((e) => setError(apiErrorMessage(e)))
  }, [])

  const summaryStats = useMemo(() => {
    if (!reports) return null
    const xray = reports.filter((r) => r.modality === 'xray').length
    const ct = reports.filter((r) => r.modality === 'ct').length
    const ctReports = reports.filter((r) => r.modality === 'ct')
    const avgCtLocations =
      ctReports.length > 0
        ? (ctReports.reduce((sum, r) => sum + r.location_count, 0) / ctReports.length).toFixed(1)
        : '—'

    const findingCounts = new Map<string, number>()
    for (const r of reports) {
      const key = (r.primary_finding ?? 'Unknown').toUpperCase()
      findingCounts.set(key, (findingCounts.get(key) ?? 0) + 1)
    }
    const findingData: BarDatum[] = [...findingCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6)
      .map(([label, value]) => ({ label, value, colorVar: FINDING_COLORS[label] ?? 'var(--color-chart-1)' }))

    const modalityData: BarDatum[] = [
      { label: 'X-RAY', value: xray, colorVar: 'var(--color-chart-1)' },
      { label: 'CT', value: ct, colorVar: 'var(--color-chart-2)' },
    ]

    const byDay = new Map<string, number>()
    for (const day of last14Days()) byDay.set(day, 0)
    for (const r of reports) {
      const day = r.created_at.slice(0, 10)
      if (byDay.has(day)) byDay.set(day, (byDay.get(day) ?? 0) + 1)
    }
    const timeData: BarDatum[] = [...byDay.entries()].map(([day, value]) => ({
      label: day.slice(5), // MM-DD
      value,
      colorVar: 'var(--color-chart-1)',
    }))

    return { findingData, modalityData, timeData, avgCtLocations }
  }, [reports])

  const deepDive = useMemo(() => {
    if (!details || details.length === 0) return null
    const providerCounts = new Map<string, number>()
    let reviewRequired = 0
    let totalCandidates = 0
    for (const report of details) {
      for (const c of report.candidates) {
        totalCandidates += 1
        providerCounts.set(c.llm_provider, (providerCounts.get(c.llm_provider) ?? 0) + 1)
        if (c.requires_professional_review) reviewRequired += 1
      }
    }
    const providerData: BarDatum[] = [...providerCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([label, value]) => ({
        label: label.toUpperCase(),
        value,
        colorVar: PROVIDER_COLORS[label] ?? 'var(--color-chart-1)',
      }))
    const reviewPct = totalCandidates > 0 ? Math.round((reviewRequired / totalCandidates) * 100) : 0
    return { providerData, reviewPct, totalCandidates, n: details.length }
  }, [details])

  return (
    <div className="p-8 max-w-6xl">
      <h1 className="text-xl font-semibold tracking-tight mb-1">Analytics</h1>
      <p className="text-sm text-ink-muted mb-6">
        Computed from real stored reports — nothing here is simulated or estimated.
      </p>

      {error && (
        <div className="rounded-lg border border-danger/40 bg-danger-soft px-4 py-2.5 text-sm text-danger mb-6">
          {error}
        </div>
      )}

      {reports === null && !error && <div className="text-sm text-ink-faint">Loading…</div>}

      {reports !== null && reports.length === 0 && (
        <div className="card text-sm text-ink-muted">No reports yet — analytics need at least one real report.</div>
      )}

      {summaryStats && reports && reports.length > 0 && (
        <div className="flex flex-col gap-5">
          <div className="grid grid-cols-3 gap-4">
            <div className="card">
              <div className="text-xs font-medium text-ink-faint uppercase tracking-wide mb-4">
                Modality Breakdown
              </div>
              <BarChart data={summaryStats.modalityData} height={120} />
            </div>
            <StatTile
              label="Avg. candidate locations per CT study"
              value={summaryStats.avgCtLocations}
              hint="Multi-nodule reports raise this above 1.0"
              accent="warn"
            />
            <StatTile
              label="Studies analyzed (last 14 days)"
              value={summaryStats.timeData.reduce((s, d) => s + d.value, 0)}
              accent="accent"
            />
          </div>

          <div className="card">
            <div className="text-xs font-medium text-ink-faint uppercase tracking-wide mb-4">
              Studies Over Time (last 14 days)
            </div>
            <BarChart data={summaryStats.timeData} height={140} />
          </div>

          <div className="card">
            <div className="text-xs font-medium text-ink-faint uppercase tracking-wide mb-4">
              Primary Finding Distribution
            </div>
            <BarChart data={summaryStats.findingData} />
          </div>

          {deepDive && (
            <div className="card">
              <div className="text-xs font-medium text-ink-faint uppercase tracking-wide mb-1">
                LLM Provider Usage &amp; Safety Pipeline
              </div>
              <p className="text-[11.5px] text-ink-faint mb-4">
                Based on the {deepDive.n} most recently generated report{deepDive.n === 1 ? '' : 's'} (
                {deepDive.totalCandidates} candidate location{deepDive.totalCandidates === 1 ? '' : 's'}) — not
                every report, kept small so this page stays fast.
              </p>
              <div className="grid grid-cols-[1fr_auto] gap-6 items-center">
                <BarChart data={deepDive.providerData} height={120} />
                <div className="text-center px-4">
                  <div className="text-3xl font-semibold text-positive">{deepDive.reviewPct}%</div>
                  <div className="text-[11px] text-ink-faint mt-1 max-w-[140px]">
                    flagged &quot;requires professional review&quot; — enforced by the safety pipeline, not
                    optional
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
