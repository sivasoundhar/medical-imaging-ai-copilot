import { useEffect, useState } from 'react'
import { apiErrorMessage, getSystemInfo } from '../api/client'
import type { SystemInfoResponse } from '../api/types'
import { applyTheme, getStoredTheme, type Theme } from '../lib/theme'
import { MoonIcon, SunIcon } from '../components/icons'

const SAFETY_PIPELINE = [
  { name: 'Groundedness validation', detail: 'Rejects any LLM-claimed finding not present in the real vision output.' },
  { name: 'Output guard', detail: 'Flags certainty language, unsupported treatment claims, missing review flag.' },
  { name: 'Input guard', detail: 'Validates question length and screens for prompt-injection phrasing.' },
  { name: 'Retry + cross-provider fallback', detail: 'Retries the primary provider, then falls back if configured.' },
]

export default function Settings() {
  const [info, setInfo] = useState<SystemInfoResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [theme, setTheme] = useState<Theme>(getStoredTheme())

  useEffect(() => {
    getSystemInfo()
      .then(setInfo)
      .catch((e) => setError(apiErrorMessage(e)))
  }, [])

  function toggleTheme() {
    const next: Theme = theme === 'dark' ? 'light' : 'dark'
    setTheme(next)
    applyTheme(next)
  }

  return (
    <div className="p-8 max-w-3xl">
      <h1 className="text-xl font-semibold tracking-tight mb-1">Settings</h1>
      <p className="text-sm text-ink-muted mb-6">
        Read-only system configuration, reflecting real environment settings — not editable knobs.
      </p>

      {error && (
        <div className="rounded-lg border border-danger/40 bg-danger-soft px-4 py-2.5 text-sm text-danger mb-6">
          {error}
        </div>
      )}

      {info === null && !error && <div className="text-sm text-ink-faint">Loading…</div>}

      {info && (
        <div className="flex flex-col gap-5">
          <div className="card">
            <div className="text-xs font-medium text-ink-faint uppercase tracking-wide mb-3">Appearance</div>
            <div className="flex items-center justify-between">
              <div className="text-sm text-ink-muted">Interface theme</div>
              <button type="button" onClick={toggleTheme} className="btn-secondary">
                {theme === 'dark' ? <SunIcon className="w-4 h-4" /> : <MoonIcon className="w-4 h-4" />}
                Switch to {theme === 'dark' ? 'light' : 'dark'}
              </button>
            </div>
          </div>

          <div className="card">
            <div className="text-xs font-medium text-ink-faint uppercase tracking-wide mb-3">LLM Configuration</div>
            <SettingsRow label="Primary provider" value={info.llm_provider.toUpperCase()} />
            <SettingsRow label="Active model" value={info.llm_model ?? 'Not configured'} mono />
            <SettingsRow
              label="Fallback provider"
              value={info.llm_fallback_provider ? info.llm_fallback_provider.toUpperCase() : 'None configured'}
            />
            {info.llm_fallback_model && <SettingsRow label="Fallback model" value={info.llm_fallback_model} mono />}
          </div>

          <div className="card">
            <div className="text-xs font-medium text-ink-faint uppercase tracking-wide mb-3">Vision Models</div>
            <ModelRow name="2D chest X-ray classifier" version={info.model_2d_version} available={info.model_2d_available} />
            <ModelRow name="3D CT nodule classifier" version={info.model_3d_version} available={info.model_3d_available} />
          </div>

          <div className="card">
            <div className="text-xs font-medium text-ink-faint uppercase tracking-wide mb-3">Safety Pipeline</div>
            <div className="space-y-3">
              {SAFETY_PIPELINE.map((s) => (
                <div key={s.name} className="flex gap-3">
                  <div className="w-1.5 h-1.5 rounded-full bg-positive mt-1.5 shrink-0" />
                  <div>
                    <div className="text-sm text-ink">{s.name}</div>
                    <div className="text-[11.5px] text-ink-faint">{s.detail}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="card">
            <div className="text-xs font-medium text-ink-faint uppercase tracking-wide mb-3">Application</div>
            <SettingsRow label="Name" value={info.app_name} />
            <SettingsRow label="Environment" value={info.app_env} />
            <SettingsRow label="API version" value={info.api_version} mono />
          </div>

          <div className="rounded-lg border border-warn-border bg-warn-soft px-4 py-3 text-[12px] leading-relaxed text-ink-muted">
            <span className="text-warn font-semibold">Research prototype.</span> No real patient data is
            processed or stored — all patient/study identifiers throughout this app are synthetic demo values.
            AI-generated output is never a diagnosis and always requires professional review.
          </div>
        </div>
      )}
    </div>
  )
}

function SettingsRow({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-center justify-between py-1.5 text-sm">
      <span className="text-ink-faint">{label}</span>
      <span className={mono ? 'font-mono-tabular text-ink' : 'text-ink'}>{value}</span>
    </div>
  )
}

function ModelRow({ name, version, available }: { name: string; version: string; available: boolean }) {
  return (
    <div className="flex items-center justify-between py-1.5 text-sm">
      <div>
        <div className="text-ink">{name}</div>
        <div className="text-[11px] text-ink-faint font-mono-tabular">{version}</div>
      </div>
      <span
        className={`text-[11px] font-medium px-2 py-1 rounded-full ${
          available ? 'text-positive bg-positive-soft' : 'text-danger bg-danger-soft'
        }`}
      >
        {available ? 'Checkpoint found' : 'Checkpoint missing'}
      </span>
    </div>
  )
}
