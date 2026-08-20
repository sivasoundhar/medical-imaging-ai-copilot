// Stat tile contract per the dataviz skill: label (sentence case, no
// trailing colon) + value (proportional figures, not tabular-nums --
// reserved for column alignment, not a standalone big number).

interface Props {
  label: string
  value: string | number
  hint?: string
  accent?: 'accent' | 'positive' | 'warn' | 'danger'
}

const ACCENT_TEXT: Record<NonNullable<Props['accent']>, string> = {
  accent: 'text-accent',
  positive: 'text-positive',
  warn: 'text-warn',
  danger: 'text-danger',
}

export default function StatTile({ label, value, hint, accent = 'accent' }: Props) {
  return (
    <div className="card">
      <div className="text-xs font-medium text-ink-faint uppercase tracking-wide mb-2">{label}</div>
      <div className={`text-3xl font-semibold ${ACCENT_TEXT[accent]}`}>{value}</div>
      {hint && <div className="mt-1.5 text-[11.5px] text-ink-faint leading-snug">{hint}</div>}
    </div>
  )
}
