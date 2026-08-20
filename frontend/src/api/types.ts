// Mirrors src/schemas/*.py exactly — field names and shapes must stay
// in sync with the backend by hand (no codegen step in this project).

export interface PatientInfo {
  patient_id: string
  name: string
  age: number
  sex: 'Male' | 'Female' | 'Other'
}

export interface StudyInfo {
  modality: 'xray' | 'ct'
  body_region: string
  study_date: string
}

export interface Finding {
  label: string
  probability: number
}

export interface VisionResult {
  findings: Finding[]
  heatmap_available: boolean
  localization: string | null
  heatmap_base64: string | null
  // CT has no Grad-CAM equivalent (heatmap_base64 is always null there) --
  // this is the analyzed slice with the candidate location marked instead,
  // NOT a saliency map. Always null for X-ray.
  location_preview_base64: string | null
  // X-ray only: the resized-to-224x224 image the model actually saw,
  // before the Grad-CAM heatmap was blended in -- identical dimensions to
  // heatmap_base64 (both come from the same resize step), so the two can
  // be shown side by side with no size mismatch and no cropping needed.
  // Always null for CT.
  resized_original_base64: string | null
}

export interface LLMResult {
  provider: string
  model: string
  report: string
  grounded: boolean
}

export interface SafetyInfo {
  requires_professional_review: boolean
}

export interface AnalysisResponse {
  analysis_id: string
  study_id: string
  patient: PatientInfo
  study: StudyInfo
  vision: VisionResult
  llm: LLMResult | null
  safety: SafetyInfo
}

export interface CTPreviewResponse {
  slice_index: number
  num_slices: number
  width: number
  height: number
  origin: [number, number, number]
  spacing: [number, number, number]
  image_base64: string
}

export interface CopilotResponse {
  summary: string
  findings: string[]
  limitations: string[]
  requires_professional_review: boolean
  provider: string
  model: string
  grounded: boolean
  kb_sources_used: string[]
}

export interface ReportSummary {
  report_id: string
  study_id: string
  patient_name: string
  modality: string
  study_date: string
  primary_finding: string | null
  location_count: number
  created_at: string
}

// One analyzed location's findings + grounded explanation, as persisted
// and redisplayed. X-ray reports always have exactly one; CT reports may
// have several (Day 11.1 -- multiple candidate locations on one scan).
export interface ReportCandidateDetail {
  localization: string | null
  vision_findings: Finding[]
  copilot_summary: string
  copilot_findings: string[]
  copilot_limitations: string[]
  requires_professional_review: boolean
  llm_provider: string
  llm_model: string
}

export interface ReportDetail extends ReportSummary {
  patient_id: string
  patient_age: number
  patient_sex: string
  body_region: string
  vision_model_version: string
  candidates: ReportCandidateDetail[]
}

// What the frontend sends per analyzed location when generating a report.
export interface ReportCandidateInput {
  analysis: AnalysisResponse
  copilot: CopilotResponse
}

export interface ApiErrorBody {
  detail: string
}

export interface SystemInfoResponse {
  app_name: string
  app_env: string
  api_version: string
  llm_provider: string
  llm_fallback_provider: string | null
  llm_model: string | null
  llm_fallback_model: string | null
  model_2d_available: boolean
  model_2d_version: string
  model_3d_available: boolean
  model_3d_version: string
}
