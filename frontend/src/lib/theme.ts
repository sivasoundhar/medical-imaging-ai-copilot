// Day 12: a real, working light/dark toggle on top of the existing
// design tokens in index.css. Reverses Day 11's deliberate "no light
// mode" call (documented there as a reading-room-appropriate choice) —
// still defaults to dark on first visit, but a user can now actually
// switch it, and the choice persists.

export type Theme = 'dark' | 'light'

const STORAGE_KEY = 'medai-theme'

export function getStoredTheme(): Theme {
  const stored = localStorage.getItem(STORAGE_KEY)
  return stored === 'light' ? 'light' : 'dark'
}

export function applyTheme(theme: Theme): void {
  const root = document.documentElement
  if (theme === 'light') {
    root.setAttribute('data-theme', 'light')
  } else {
    root.removeAttribute('data-theme')
  }
  localStorage.setItem(STORAGE_KEY, theme)
}
