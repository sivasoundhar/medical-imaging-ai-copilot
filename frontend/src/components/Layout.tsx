import { type ReactNode, useEffect, useRef, useState } from 'react'
import { NavLink, useLocation, useNavigate } from 'react-router-dom'
import { listReports } from '../api/client'
import type { ReportSummary } from '../api/types'
import {
  BellIcon,
  ChartIcon,
  ChatIcon,
  DashboardIcon,
  ExitIcon,
  HistoryIcon,
  MoonIcon,
  SettingsIcon,
  SunIcon,
  TableIcon,
  UploadIcon,
} from './icons'
import { applyTheme, getStoredTheme, type Theme } from '../lib/theme'

const navItems = [
  { to: '/dashboard', label: 'Dashboard', icon: DashboardIcon },
  { to: '/new-study', label: 'Analyze Images', icon: UploadIcon },
  { to: '/all-studies', label: 'All Studies', icon: TableIcon },
  { to: '/copilot', label: 'AI Copilot', icon: ChatIcon },
  { to: '/reports', label: 'Reports History', icon: HistoryIcon },
  { to: '/analytics', label: 'Analytics', icon: ChartIcon },
  { to: '/settings', label: 'Settings', icon: SettingsIcon },
]

const PAGE_TITLES: Record<string, string> = {
  '/dashboard': 'Dashboard',
  '/new-study': 'Analyze Images',
  '/all-studies': 'All Studies',
  '/copilot': 'AI Copilot',
  '/reports': 'Reports History',
  '/analytics': 'Analytics',
  '/settings': 'Settings',
}

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-screen bg-bg text-ink">
      <aside className="w-60 shrink-0 border-r border-border bg-surface flex flex-col">
        <div className="px-5 py-6 border-b border-border">
          <div className="flex items-center gap-2.5">
            <LogoMark />
            <div>
              <div className="font-semibold tracking-tight leading-tight">MedAI Copilot</div>
              <div className="text-[11px] text-ink-faint font-mono-tabular tracking-wide">
                RADIOLOGY COMMAND CENTER
              </div>
            </div>
          </div>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1">
          {navItems.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-accent-soft text-accent border border-accent/40'
                    : 'text-ink-muted hover:bg-surface-2 hover:text-ink'
                }`
              }
            >
              <Icon className="w-4 h-4 shrink-0" />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="px-4 py-4 border-t border-border">
          <div className="rounded-lg border border-warn-border bg-warn-soft px-3 py-2.5 text-[11.5px] leading-snug text-ink-muted">
            <span className="text-warn font-semibold">Research prototype.</span> AI-generated
            output only — not a diagnostic device. Requires professional review.
          </div>
        </div>
      </aside>

      <div className="flex-1 min-w-0 flex flex-col">
        <TopBar />
        <main className="flex-1 min-w-0">{children}</main>
      </div>
    </div>
  )
}

function TopBar() {
  const [theme, setTheme] = useState<Theme>(getStoredTheme())
  const location = useLocation()
  const title =
    PAGE_TITLES[location.pathname] ??
    (location.pathname.startsWith('/reports/') ? 'Report Detail' : 'MedAI Copilot')

  function toggleTheme() {
    const next: Theme = theme === 'dark' ? 'light' : 'dark'
    setTheme(next)
    applyTheme(next)
  }

  return (
    <header className="h-14 shrink-0 border-b border-border bg-surface flex items-center justify-between px-6">
      <div className="text-sm font-semibold text-ink">{title}</div>
      <div className="flex items-center gap-1.5">
        <NotificationBell />
        <button
          type="button"
          onClick={toggleTheme}
          title={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
          className="w-8 h-8 rounded-md flex items-center justify-center text-ink-muted hover:bg-surface-2 hover:text-ink transition-colors"
        >
          {theme === 'dark' ? <SunIcon className="w-4 h-4" /> : <MoonIcon className="w-4 h-4" />}
        </button>
        <ExitButton />
      </div>
    </header>
  )
}

function NotificationBell() {
  const [open, setOpen] = useState(false)
  const [recent, setRecent] = useState<ReportSummary[] | null>(null)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onClickOutside)
    return () => document.removeEventListener('mousedown', onClickOutside)
  }, [])

  async function toggle() {
    const next = !open
    setOpen(next)
    if (next && recent === null) {
      try {
        setRecent(await listReports(5))
      } catch {
        setRecent([])
      }
    }
  }

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        onClick={toggle}
        title="Recent report activity"
        className="w-8 h-8 rounded-md flex items-center justify-center text-ink-muted hover:bg-surface-2 hover:text-ink transition-colors"
      >
        <BellIcon className="w-4 h-4" />
      </button>
      {open && (
        <div className="absolute right-0 top-10 w-72 rounded-lg border border-border bg-surface shadow-xl z-20 overflow-hidden">
          <div className="px-3 py-2.5 border-b border-border text-xs font-medium text-ink-faint uppercase tracking-wide">
            Recent Reports
          </div>
          {recent === null ? (
            <div className="px-3 py-4 text-xs text-ink-faint">Loading…</div>
          ) : recent.length === 0 ? (
            <div className="px-3 py-4 text-xs text-ink-faint">No reports generated yet.</div>
          ) : (
            <ul>
              {recent.map((r) => (
                <li key={r.report_id} className="border-b border-border last:border-0">
                  <NavLink
                    to={`/reports/${r.report_id}`}
                    className="block px-3 py-2.5 hover:bg-surface-2 transition-colors"
                    onClick={() => setOpen(false)}
                  >
                    <div className="text-xs text-ink font-medium truncate">
                      {r.patient_name} — {r.primary_finding ?? 'No finding'}
                    </div>
                    <div className="text-[10.5px] text-ink-faint font-mono-tabular mt-0.5">
                      {r.modality.toUpperCase()} · {new Date(r.created_at).toLocaleString()}
                    </div>
                  </NavLink>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}

function ExitButton() {
  const navigate = useNavigate()
  return (
    <button
      type="button"
      title="Reset session (no authentication in this prototype) — returns to Dashboard"
      onClick={() => navigate('/dashboard')}
      className="w-8 h-8 rounded-md flex items-center justify-center text-ink-muted hover:bg-danger-soft hover:text-danger transition-colors"
    >
      <ExitIcon className="w-4 h-4" />
    </button>
  )
}

function LogoMark() {
  return (
    <div className="w-8 h-8 rounded-md bg-accent-soft border border-accent/40 flex items-center justify-center">
      <svg viewBox="0 0 24 24" className="w-4.5 h-4.5 text-accent" fill="none" stroke="currentColor" strokeWidth="1.8">
        <path d="M12 3v18M4 8h16M4 16h16" strokeLinecap="round" />
        <circle cx="12" cy="12" r="2.2" fill="currentColor" stroke="none" />
      </svg>
    </div>
  )
}
