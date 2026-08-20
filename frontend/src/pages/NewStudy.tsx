import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { analyzeCt, analyzeXray, apiErrorMessage, generateReport, type NewStudyPatientInfo } from '../api/client'
import type { AnalysisResponse, CopilotResponse, ReportCandidateInput } from '../api/types'
import PatientForm from '../components/PatientForm'
import XrayUpload from '../components/XrayUpload'
import CtUploadAndPicker from '../components/CtUploadAndPicker'
import AnalysisResult from '../components/AnalysisResult'
import CopilotPanel from '../components/CopilotPanel'
import { Spinner } from '../components/XrayUpload'

type Step = 'patient' | 'upload' | 'analysis'

// Real bug found live: this used to be one combined string sent for
// every report regardless of modality, so a pure-CT report's "Vision
// model version" metadata claimed the 2D X-ray model was involved too
// (and vice versa) -- misleading, since only one model ever actually
// runs per report. Keyed per modality so the report only ever names the
// model that actually produced its findings.
const VISION_MODEL_VERSIONS: Record<'xray' | 'ct', string> = {
  xray: 'resnet50-2d-v1',
  ct: 'nodule-3d-cnn-v1',
}

export default function NewStudy() {
  const navigate = useNavigate()
  const [step, setStep] = useState<Step>('patient')
  const [patient, setPatient] = useState<NewStudyPatientInfo | null>(null)
  const [modality, setModality] = useState<'xray' | 'ct'>('xray')
  const [analyzing, setAnalyzing] = useState(false)
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null)
  const [copilotResult, setCopilotResult] = useState<CopilotResponse | null>(null)
  // CT only: locations already analyzed + explained, waiting to go into
  // one combined report (Day 11.1) -- e.g. checking several candidate
  // nodules on one scan before generating a single report covering all
  // of them, matching how a real CT read can cover multiple nodules.
  const [savedCandidates, setSavedCandidates] = useState<ReportCandidateInput[]>([])
  const [error, setError] = useState<string | null>(null)
  const [generatingReport, setGeneratingReport] = useState(false)

  function handlePatientSubmit(p: NewStudyPatientInfo, m: 'xray' | 'ct') {
    setPatient(p)
    setModality(m)
    setStep('upload')
  }

  async function handleXrayAnalyze(file: File) {
    if (!patient) return
    setAnalyzing(true)
    setError(null)
    try {
      const result = await analyzeXray(file, patient)
      setAnalysis(result)
      setStep('analysis')
    } catch (e) {
      setError(apiErrorMessage(e))
    } finally {
      setAnalyzing(false)
    }
  }

  async function handleCtAnalyze(mhdFile: File, rawFile: File, coord: [number, number, number]) {
    if (!patient) return
    setAnalyzing(true)
    setError(null)
    try {
      const result = await analyzeCt(mhdFile, rawFile, coord, patient)
      setAnalysis(result)
      setCopilotResult(null) // a new analysis needs its own grounded explanation, never the last one's
      setStep('analysis') // purely advances the step-header chip -- CT renders the same screen either way
    } catch (e) {
      setError(apiErrorMessage(e))
    } finally {
      setAnalyzing(false)
    }
  }

  function handleSaveLocation() {
    if (!analysis || !copilotResult) return
    setSavedCandidates((prev) => [...prev, { analysis, copilot: copilotResult }])
    // Back to the placeholder, picker's own state (files, slice preview)
    // stays intact -- ready to pick and analyze the next location.
    setAnalysis(null)
    setCopilotResult(null)
  }

  function handleRemoveSavedLocation(index: number) {
    setSavedCandidates((prev) => prev.filter((_, i) => i !== index))
  }

  async function handleGenerateReport() {
    // CT: everything saved so far, plus the currently-viewed location if
    // it hasn't been explicitly saved yet -- so a single-location report
    // still only takes one click, same as the X-ray flow.
    const candidates: ReportCandidateInput[] =
      modality === 'ct'
        ? [...savedCandidates, ...(analysis && copilotResult ? [{ analysis, copilot: copilotResult }] : [])]
        : analysis && copilotResult
          ? [{ analysis, copilot: copilotResult }]
          : []
    if (candidates.length === 0) return

    setGeneratingReport(true)
    setError(null)
    try {
      const summary = await generateReport(candidates, VISION_MODEL_VERSIONS[modality])
      navigate(`/reports/${summary.report_id}`)
    } catch (e) {
      setError(apiErrorMessage(e))
    } finally {
      setGeneratingReport(false)
    }
  }

  return (
    <div className="p-8 max-w-[1600px]">
      <StepHeader step={step} />

      {error && (
        <div className="mb-5 rounded-lg border border-danger/40 bg-danger-soft px-4 py-2.5 text-sm text-danger max-w-xl">
          {error}
        </div>
      )}

      {step === 'patient' && <PatientForm onSubmit={handlePatientSubmit} />}

      {step === 'upload' && modality === 'xray' && (
        <XrayUpload onAnalyze={handleXrayAnalyze} loading={analyzing} />
      )}

      {step === 'analysis' && modality === 'xray' && analysis && (
        <div>
          <div className="grid grid-cols-[1fr_360px] gap-5 items-start">
            <AnalysisResult analysis={analysis} />
            <CopilotPanel key={analysis.analysis_id} analysis={analysis} onCopilotResult={setCopilotResult} />
          </div>
          <ReportButton
            disabled={!copilotResult || generatingReport}
            generating={generatingReport}
            label="Generate Professional Report"
            hint="Ask the Copilot something first — a report needs a grounded explanation to include."
            onClick={handleGenerateReport}
          />
        </div>
      )}

      {/* CT: picker (left) and result (right) live side by side, permanently
          -- picking a new point and re-analyzing never navigates away, so
          checking several candidate locations on one scan needs no
          re-upload. Covers both 'upload' and 'analysis' steps identically;
          only the right-hand panel's content changes once a result exists. */}
      {modality === 'ct' && step !== 'patient' && (
        <div className="grid grid-cols-[1fr_520px] gap-5 items-start">
          <CtUploadAndPicker onAnalyze={handleCtAnalyze} loading={analyzing} />

          <div className="flex flex-col gap-5">
            {analysis ? (
              <>
                <AnalysisResult analysis={analysis} />
                {/* Keyed on analysis_id so picking a second candidate starts
                    a clean Copilot chat -- never shows a prior candidate's
                    explanation next to this candidate's findings. */}
                <CopilotPanel key={analysis.analysis_id} analysis={analysis} onCopilotResult={setCopilotResult} />
                <div className="flex items-center gap-3">
                  <button className="btn-secondary" disabled={!copilotResult} onClick={handleSaveLocation}>
                    Save This Location &amp; Pick Another
                  </button>
                  {!copilotResult && (
                    <span className="text-xs text-ink-faint">Ask the Copilot first to save this location.</span>
                  )}
                </div>
              </>
            ) : (
              <div className="card flex items-center justify-center min-h-[200px] text-center text-sm text-ink-faint">
                Pick a point on the scan and click "Analyze This Location" —
                the result will appear here.
              </div>
            )}

            {savedCandidates.length > 0 && (
              <div className="card">
                <div className="text-xs font-medium text-ink-faint uppercase tracking-wide mb-2.5">
                  Saved Locations ({savedCandidates.length})
                </div>
                <div className="space-y-1.5">
                  {savedCandidates.map((c, i) => {
                    const top = c.analysis.vision.findings[0]
                    return (
                      <div
                        key={i}
                        className="flex items-center justify-between rounded-md bg-surface-2 px-3 py-2 text-sm"
                      >
                        <span>
                          {top?.label ?? '—'}
                          {top && (
                            <span className="ml-2 font-mono-tabular text-ink-faint">
                              {/* Clamp + 2dp, same fix as Day 11's report PDF
                                  (_format_probability) -- toFixed(1) alone can
                                  round a real 0.99998 up to a false "100.0%". */}
                              {Math.min(top.probability * 100, 99.99).toFixed(2)}%
                            </span>
                          )}
                        </span>
                        <button
                          className="text-xs text-ink-faint hover:text-danger"
                          onClick={() => handleRemoveSavedLocation(i)}
                        >
                          Remove
                        </button>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            <ReportButton
              disabled={(savedCandidates.length === 0 && !(analysis && copilotResult)) || generatingReport}
              generating={generatingReport}
              label={(() => {
                const count = savedCandidates.length + (analysis && copilotResult ? 1 : 0)
                return count > 1 ? `Generate Professional Report (${count} locations)` : 'Generate Professional Report'
              })()}
              hint="Save at least one analyzed location first."
              onClick={handleGenerateReport}
            />
          </div>
        </div>
      )}
    </div>
  )
}

function ReportButton({
  disabled,
  generating,
  label,
  hint,
  onClick,
}: {
  disabled: boolean
  generating: boolean
  label: string
  hint: string
  onClick: () => void
}) {
  return (
    <div className="mt-6 flex items-center gap-3">
      <button className="btn-primary" disabled={disabled} onClick={onClick}>
        {generating ? <Spinner /> : null}
        {generating ? 'Generating PDF…' : label}
      </button>
      {disabled && !generating && <span className="text-xs text-ink-faint">{hint}</span>}
    </div>
  )
}

function StepHeader({ step }: { step: Step }) {
  const steps: { key: Step; label: string }[] = [
    { key: 'patient', label: 'Patient & Study' },
    { key: 'upload', label: 'Upload' },
    { key: 'analysis', label: 'Analysis & Copilot' },
  ]
  const activeIndex = steps.findIndex((s) => s.key === step)

  return (
    <div className="mb-8">
      <h1 className="text-xl font-semibold tracking-tight mb-4">New Study</h1>
      <div className="flex items-center gap-2 text-xs font-medium">
        {steps.map((s, i) => (
          <div key={s.key} className="flex items-center gap-2">
            <span
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full ${
                i <= activeIndex ? 'text-accent bg-accent-soft border border-accent/30' : 'text-ink-faint'
              }`}
            >
              <span className="font-mono-tabular">{i + 1}</span>
              {s.label}
            </span>
            {i < steps.length - 1 && <span className="text-ink-faint">→</span>}
          </div>
        ))}
      </div>
    </div>
  )
}
