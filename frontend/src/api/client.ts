import axios, { AxiosError } from 'axios'
import type {
  AnalysisResponse,
  ApiErrorBody,
  CopilotResponse,
  CTPreviewResponse,
  Finding,
  ReportCandidateInput,
  ReportDetail,
  ReportSummary,
  SystemInfoResponse,
} from './types'

// Vite's dev-server proxy (vite.config.ts) forwards /api/* to the real
// FastAPI backend, so relative paths work in both dev and a production
// build served behind the same origin/reverse proxy.
const API_BASE = '/api/v1'

const http = axios.create({ baseURL: API_BASE })

/** Every backend error response is `{"detail": "..."}` (src/schemas/api.py
 * ErrorResponse) — normalize axios errors down to that message string so
 * callers don't need to know axios's error shape. */
export function apiErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const err = error as AxiosError<ApiErrorBody>
    return err.response?.data?.detail ?? err.message
  }
  return String(error)
}

export interface NewStudyPatientInfo {
  patient_id: string
  patient_name: string
  patient_age: number
  patient_sex: 'Male' | 'Female' | 'Other'
  body_region: string
  study_date?: string
  study_id?: string
}

export async function analyzeXray(file: File, patient: NewStudyPatientInfo): Promise<AnalysisResponse> {
  const form = new FormData()
  form.append('modality', 'xray')
  form.append('file', file)
  appendPatientFields(form, patient)
  const { data } = await http.post<AnalysisResponse>('/imaging/analyze', form)
  return data
}

export async function analyzeCt(
  mhdFile: File,
  rawFile: File,
  coord: [number, number, number],
  patient: NewStudyPatientInfo,
): Promise<AnalysisResponse> {
  const form = new FormData()
  form.append('modality', 'ct')
  form.append('file', mhdFile)
  form.append('raw_file', rawFile)
  form.append('coord_x', String(coord[0]))
  form.append('coord_y', String(coord[1]))
  form.append('coord_z', String(coord[2]))
  appendPatientFields(form, patient)
  const { data } = await http.post<AnalysisResponse>('/imaging/analyze', form)
  return data
}

export async function ctPreview(
  mhdFile: File,
  rawFile: File,
  sliceIndex?: number,
): Promise<CTPreviewResponse> {
  const form = new FormData()
  form.append('file', mhdFile)
  form.append('raw_file', rawFile)
  if (sliceIndex !== undefined) form.append('slice_index', String(sliceIndex))
  const { data } = await http.post<CTPreviewResponse>('/imaging/ct-preview', form)
  return data
}

export async function copilotReport(
  findings: Finding[],
  localization: string | null,
  modality: 'xray' | 'ct',
): Promise<CopilotResponse> {
  const { data } = await http.post<CopilotResponse>('/copilot/report', {
    findings,
    localization,
    modality,
  })
  return data
}

export async function copilotAsk(
  question: string,
  findings: Finding[],
  localization: string | null,
  modality: 'xray' | 'ct',
  conversationHistory?: { role: string; content: string }[],
): Promise<CopilotResponse> {
  const { data } = await http.post<CopilotResponse>('/copilot/ask', {
    question,
    findings,
    localization,
    modality,
    conversation_history: conversationHistory ?? null,
  })
  return data
}

export async function generateReport(
  candidates: ReportCandidateInput[],
  visionModelVersion: string,
): Promise<ReportSummary> {
  const { data } = await http.post<ReportSummary>('/report/generate', {
    candidates,
    vision_model_version: visionModelVersion,
  })
  return data
}

export async function listReports(limit = 100): Promise<ReportSummary[]> {
  const { data } = await http.get<ReportSummary[]>('/reports', { params: { limit } })
  return data
}

export async function getReport(reportId: string): Promise<ReportDetail> {
  const { data } = await http.get<ReportDetail>(`/reports/${reportId}`)
  return data
}

export function reportPdfUrl(reportId: string): string {
  return `${API_BASE}/reports/${reportId}/pdf`
}

export async function deleteReport(reportId: string): Promise<void> {
  await http.delete(`/reports/${reportId}`)
}

export async function getSystemInfo(): Promise<SystemInfoResponse> {
  const { data } = await http.get<SystemInfoResponse>('/system/info')
  return data
}

function appendPatientFields(form: FormData, patient: NewStudyPatientInfo): void {
  form.append('patient_id', patient.patient_id)
  form.append('patient_name', patient.patient_name)
  form.append('patient_age', String(patient.patient_age))
  form.append('patient_sex', patient.patient_sex)
  form.append('body_region', patient.body_region)
  if (patient.study_date) form.append('study_date', patient.study_date)
  if (patient.study_id) form.append('study_id', patient.study_id)
}
