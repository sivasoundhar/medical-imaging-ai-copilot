import { useState } from 'react'

// Simple column chart built in plain HTML/CSS (no charting library — kept
// small since the app only needs a handful of straightforward category
// breakdowns). Follows the dataviz skill's mark spec: bars capped at
// 24px thick, 4px rounded data-end / square baseline, a 2px surface gap
// between bars, direct value labels at the cap, and a per-bar hover
// tooltip (never relying on color alone -- every bar already carries a
// text category label underneath it).

export interface BarDatum {
  label: string
  value: number
  colorVar: string // e.g. 'var(--color-chart-1)'
}

export default function BarChart({
  data,
  height = 160,
  formatValue = (v: number) => String(v),
}: {
  data: BarDatum[]
  height?: number
  formatValue?: (v: number) => string
}) {
  const [hovered, setHovered] = useState<number | null>(null)
  const max = Math.max(1, ...data.map((d) => d.value))

  if (data.length === 0) {
    return <div className="text-sm text-ink-faint py-6 text-center">No data yet.</div>
  }

  return (
    <div className="flex items-end gap-2" style={{ height }}>
      {data.map((d, i) => {
        const barHeightPct = (d.value / max) * 100
        const isHovered = hovered === i
        return (
          <div
            key={d.label}
            className="relative flex-1 min-w-0 flex flex-col items-center justify-end h-full"
            onMouseEnter={() => setHovered(i)}
            onMouseLeave={() => setHovered(null)}
          >
            {isHovered && (
              <div className="absolute -top-1 -translate-y-full rounded-md border border-border bg-surface-2 px-2 py-1 text-[11px] font-mono-tabular text-ink shadow-lg whitespace-nowrap z-10">
                {d.label}: {formatValue(d.value)}
              </div>
            )}
            <div className="text-[11px] font-mono-tabular text-ink-muted mb-1">{formatValue(d.value)}</div>
            <div
              className="w-full rounded-t-[4px] transition-opacity"
              style={{
                height: `${Math.max(barHeightPct, 2)}%`,
                maxWidth: 40,
                background: d.colorVar,
                opacity: isHovered ? 1 : 0.85,
                marginInline: 'auto',
              }}
            />
            <div className="mt-2 text-[10.5px] text-ink-faint text-center leading-normal truncate w-full pb-0.5">
              {d.label}
            </div>
          </div>
        )
      })}
    </div>
  )
}
