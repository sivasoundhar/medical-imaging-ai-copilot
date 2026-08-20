import type { AnalysisResponse } from '../api/types'
import CompareSlider from './CompareSlider'

function isNormalLabel(label: string): boolean {
  const l = label.toLowerCase()
  return l.includes('normal') || l.includes('non_nodule') || l.includes('non-nodule')
}

// Matches the Grad-CAM colormap's low->high range (see gradcam.py) --
// purely a visual legend, not computed from real per-pixel data.
const GRADCAM_LEGEND_GRADIENT =
  'linear-gradient(to right, #1e3a8a, #0ea5e9, #22c55e, #eab308, #f97316, #dc2626)'

export default function AnalysisResult({ analysis }: { analysis: AnalysisResponse }) {
  return (
    <div className="flex flex-col gap-5">
      <div className="card flex flex-col">
        <div className="text-xs font-medium text-ink-faint uppercase tracking-wide mb-3">
          {analysis.vision.heatmap_base64
            ? 'Study Overview'
            : analysis.study.modality === 'ct'
              ? 'Candidate Location Preview'
              : 'Grad-CAM Explainability'}
        </div>
        {analysis.vision.heatmap_base64 ? (
          <>
            {analysis.vision.resized_original_base64 ? (
              // resized_original_base64 is the exact same resize-to-224x224
              // the model ran Grad-CAM on (see VisionResult's docstring in
              // schemas/imaging.py) -- identical dimensions to
              // heatmap_base64 by construction, so the drag-to-compare
              // slider never letterboxes or crops either layer. (Earlier
              // attempts at a static side-by-side tried the raw uploaded
              // file instead and hit real size-mismatch bugs -- see git
              // history / PROGRESS_LOG.md Day 11.1 items 17/19/20/21 --
              // the real fix was using the same image data, not a CSS trick.)
              <CompareSlider
                originalSrc={`data:image/png;base64,${analysis.vision.resized_original_base64}`}
                overlaySrc={`data:image/png;base64,${analysis.vision.heatmap_base64}`}
              />
            ) : (
              <figure>
                <div className="aspect-square w-full rounded-lg border border-border bg-black/20 overflow-hidden">
                  <img
                    src={`data:image/png;base64,${analysis.vision.heatmap_base64}`}
                    alt="Grad-CAM heatmap overlay"
                    className="w-full h-full object-contain"
                  />
                </div>
                <figcaption className="mt-1.5 text-[11px] text-ink-faint text-center">
                  Grad-CAM Overlay
                </figcaption>
              </figure>
            )}

            <div className="mt-4 pt-4 border-t border-border">
              <div className="text-xs font-medium text-ink-faint uppercase tracking-wide mb-1">
                Grad-CAM Heatmap
              </div>
              <p className="text-[11.5px] text-ink-faint leading-relaxed mb-2.5">
                Red areas indicate regions that most influenced the model's prediction. Does not prove a
                disease is present and is not a diagnosis.
              </p>
              <div className="h-2 rounded-full" style={{ background: GRADCAM_LEGEND_GRADIENT }} />
              <div className="mt-1.5 flex justify-between text-[10.5px] text-ink-faint">
                <span>Low Importance</span>
                <span>High Importance</span>
              </div>
            </div>
          </>
        ) : analysis.vision.location_preview_base64 ? (
          <>
            <img
              src={`data:image/png;base64,${analysis.vision.location_preview_base64}`}
              alt="CT slice with candidate location marked"
              className="rounded-lg border border-border w-full object-contain"
            />
            <p className="mt-3 text-[11.5px] text-ink-faint leading-relaxed">
              The analyzed CT slice nearest the candidate coordinate, with the location marked. Not a
              saliency map — Grad-CAM visualization has only been implemented for the 2D X-ray model.
            </p>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-sm text-ink-faint text-center py-8">
            {analysis.study.modality === 'ct'
              ? 'Preview rendering failed for this candidate. No saliency overlay either — Grad-CAM visualization has only been implemented for the 2D X-ray model.'
              : 'No Grad-CAM heatmap was generated for this analysis.'}
          </div>
        )}
      </div>

      <div className="card">
        <div className="text-xs font-medium text-ink-faint uppercase tracking-wide mb-3">AI Findings</div>

        {analysis.vision.findings.length === 0 ? (
          <div className="text-sm text-ink-muted">No findings flagged.</div>
        ) : (
          // The vision pipeline always returns exactly one Finding (the
          // model's top prediction, see imaging_service.py) -- shown as
          // the mockup's single "Primary Finding" rather than a list.
          // (A multi-label "Other Findings" sub-panel like the mockup's
          // was deliberately left out: it would need a real multi-label
          // model this project doesn't have -- see docs/MODEL_CARD.md.)
          <>
            <div className="text-[11px] font-medium text-ink-faint uppercase tracking-wide mb-1">
              Primary Finding
            </div>
            <div
              className={`text-2xl font-semibold mb-1 ${
                isNormalLabel(analysis.vision.findings[0].label) ? 'text-positive' : 'text-warn'
              }`}
            >
              {analysis.vision.findings[0].label}
            </div>
            <div className="font-mono-tabular text-sm text-ink-muted">
              {Math.min(analysis.vision.findings[0].probability * 100, 99.99).toFixed(2)}% Confidence
            </div>
            <div className="mt-2.5 h-1.5 rounded-full bg-surface-2 overflow-hidden">
              <div
                className={`h-full rounded-full ${
                  isNormalLabel(analysis.vision.findings[0].label) ? 'bg-positive' : 'bg-warn'
                }`}
                style={{ width: `${Math.min(analysis.vision.findings[0].probability * 100, 100)}%` }}
              />
            </div>
          </>
        )}

        {analysis.vision.localization && (
          <div className="mt-4 pt-4 border-t border-border">
            <div className="text-xs font-medium text-ink-faint uppercase tracking-wide mb-1">
              Localization
            </div>
            <div className="text-sm font-mono-tabular text-ink-muted">{analysis.vision.localization}</div>
          </div>
        )}
      </div>

      <div className="card">
        <div className="text-xs font-medium text-ink-faint uppercase tracking-wide mb-3">Study Info</div>
        <div className="space-y-2 text-xs">
          <InfoRow label="Study ID" value={analysis.study_id} mono />
          <InfoRow label="Patient" value={`${analysis.patient.name} (${analysis.patient.age}/${analysis.patient.sex})`} />
          <InfoRow label="Modality" value={`${analysis.study.modality.toUpperCase()} · ${analysis.study.body_region}`} />
          <InfoRow label="Date" value={analysis.study.study_date || '—'} />
          <InfoRow label="Analysis ID" value={analysis.analysis_id.slice(0, 8)} mono />
        </div>
      </div>
    </div>
  )
}

function InfoRow({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-ink-faint shrink-0">{label}</span>
      <span className={`text-right truncate ${mono ? 'font-mono-tabular text-ink-muted' : 'text-ink-muted'}`}>
        {value}
      </span>
    </div>
  )
}
