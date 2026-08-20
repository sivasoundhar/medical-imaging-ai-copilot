import { useState } from 'react'
import type { NewStudyPatientInfo } from '../api/client'

interface Props {
  onSubmit: (patient: NewStudyPatientInfo, modality: 'xray' | 'ct') => void
}

// Synthetic demo data only, per PROJECT_SPEC.md Section 26 — never real
// patient information. Pre-filled with a placeholder so the form is
// fast to submit in a demo, but every field is editable.
export default function PatientForm({ onSubmit }: Props) {
  const [patientId, setPatientId] = useState('PT-2026-00124')
  const [studyId, setStudyId] = useState('STU-2026-00089')
  const [name, setName] = useState('Jane Doe')
  const [age, setAge] = useState(45)
  const [sex, setSex] = useState<'Male' | 'Female' | 'Other'>('Female')
  const [bodyRegion, setBodyRegion] = useState('chest')
  const [modality, setModality] = useState<'xray' | 'ct'>('xray')

  return (
    <form
      className="max-w-xl"
      onSubmit={(e) => {
        e.preventDefault()
        onSubmit(
          {
            patient_id: patientId,
            patient_name: name,
            patient_age: age,
            patient_sex: sex,
            body_region: bodyRegion,
            // Empty is allowed -- the backend falls back to a generated ID
            // rather than rejecting the request -- but pass it through when
            // the user typed one so a real-looking value round-trips into
            // the report instead of a fallback.
            study_id: studyId.trim() || undefined,
          },
          modality,
        )
      }}
    >
      <div className="grid grid-cols-2 gap-4">
        <Field label="Patient ID">
          <input className="input" value={patientId} onChange={(e) => setPatientId(e.target.value)} required />
        </Field>
        <Field label="Study ID">
          <input className="input" value={studyId} onChange={(e) => setStudyId(e.target.value)} />
        </Field>
        <Field label="Name (synthetic)">
          <input className="input" value={name} onChange={(e) => setName(e.target.value)} required />
        </Field>
        <Field label="Age">
          <input
            className="input"
            type="number"
            min={0}
            max={130}
            value={age}
            onChange={(e) => setAge(Number(e.target.value))}
            required
          />
        </Field>
        <Field label="Sex">
          <select className="input" value={sex} onChange={(e) => setSex(e.target.value as typeof sex)}>
            <option>Female</option>
            <option>Male</option>
            <option>Other</option>
          </select>
        </Field>
        <Field label="Body Region">
          <input className="input" value={bodyRegion} onChange={(e) => setBodyRegion(e.target.value)} required />
        </Field>
        <Field label="Modality">
          <div className="flex gap-2 pt-0.5">
            <ModalityOption label="X-ray (2D)" active={modality === 'xray'} onClick={() => setModality('xray')} />
            <ModalityOption label="CT (3D)" active={modality === 'ct'} onClick={() => setModality('ct')} />
          </div>
        </Field>
      </div>

      <p className="mt-4 text-xs text-ink-faint leading-relaxed">
        Patient details are synthetic demo data for this portfolio project — never real patient
        information.
      </p>

      <button type="submit" className="btn-primary mt-5">
        Continue to Upload
      </button>
    </form>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="block text-xs font-medium text-ink-muted mb-1.5">{label}</span>
      {children}
    </label>
  )
}

function ModalityOption({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex-1 rounded-lg border px-3 py-2 text-sm font-medium transition-colors ${
        active
          ? 'border-accent bg-accent-soft text-accent'
          : 'border-border bg-surface-2 text-ink-muted hover:text-ink'
      }`}
    >
      {label}
    </button>
  )
}
