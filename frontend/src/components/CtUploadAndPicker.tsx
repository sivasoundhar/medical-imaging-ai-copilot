import { useEffect, useRef, useState } from 'react'
import { apiErrorMessage, ctPreview } from '../api/client'
import type { CTPreviewResponse } from '../api/types'
import { Spinner } from './XrayUpload'

interface Props {
  onAnalyze: (mhdFile: File, rawFile: File, coord: [number, number, number]) => void
  loading: boolean
}

// Mirrors src/vision/ct_preview.py's pixel_to_world() EXACTLY — same
// axis-aligned-direction assumption, same formula. If one changes, the
// other must too (see that module's docstring for why this assumption
// is safe for LUNA16 data).
function pixelToWorld(
  row: number,
  col: number,
  sliceIndex: number,
  origin: [number, number, number],
  spacing: [number, number, number],
): [number, number, number] {
  return [origin[0] + col * spacing[0], origin[1] + row * spacing[1], origin[2] + sliceIndex * spacing[2]]
}

export default function CtUploadAndPicker({ onAnalyze, loading }: Props) {
  const [mhdFile, setMhdFile] = useState<File | null>(null)
  const [rawFile, setRawFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<CTPreviewResponse | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [picked, setPicked] = useState<{ row: number; col: number; world: [number, number, number] } | null>(null)
  const imgRef = useRef<HTMLImageElement>(null)

  // Slider's own live position, separate from `preview.slice_index` --
  // dragging moves this instantly (no network call); only releasing the
  // slider fetches the new slice. `preview.image_base64` re-uploads the
  // whole .mhd/.raw pair (real CT scans run tens of MB), so firing on
  // every drag tick made scrubbing effectively unusable.
  const [sliderValue, setSliderValue] = useState(0)
  useEffect(() => {
    if (preview) setSliderValue(preview.slice_index)
  }, [preview])
  const requestIdRef = useRef(0)

  function handleFiles(files: FileList | null) {
    if (!files) return
    for (const f of Array.from(files)) {
      if (f.name.toLowerCase().endsWith('.mhd')) setMhdFile(f)
      else if (f.name.toLowerCase().endsWith('.raw')) setRawFile(f)
    }
  }

  async function loadPreview(sliceIndex?: number) {
    if (!mhdFile || !rawFile) return
    // The slider no longer disables itself while a request is in flight
    // (that's what made scrubbing feel frozen), so a quick release-drag-
    // release can fire a second request before the first resolves. Guard
    // against the slower, older one overwriting the newer result.
    const requestId = ++requestIdRef.current
    setPreviewLoading(true)
    setError(null)
    try {
      const result = await ctPreview(mhdFile, rawFile, sliceIndex)
      if (requestId !== requestIdRef.current) return // superseded by a newer request
      setPreview(result)
      setPicked(null)
    } catch (e) {
      if (requestId !== requestIdRef.current) return
      setError(apiErrorMessage(e))
    } finally {
      if (requestId === requestIdRef.current) setPreviewLoading(false)
    }
  }

  function handleImageClick(e: React.MouseEvent<HTMLImageElement>) {
    if (!preview || !imgRef.current) return
    const rect = imgRef.current.getBoundingClientRect()
    // Scale from the displayed (possibly resized) element back to the
    // image's real pixel dimensions before converting to world coords.
    const scaleX = preview.width / rect.width
    const scaleY = preview.height / rect.height
    const col = Math.round((e.clientX - rect.left) * scaleX)
    const row = Math.round((e.clientY - rect.top) * scaleY)
    const world = pixelToWorld(row, col, preview.slice_index, preview.origin, preview.spacing)
    setPicked({ row, col, world })
  }

  const bothFilesSelected = mhdFile && rawFile

  return (
    <div className="max-w-3xl">
      {!preview && (
        <>
          <label
            className="flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed
              border-border bg-surface-2 px-6 py-10 text-center cursor-pointer transition-colors
              hover:border-accent/60"
          >
            <input type="file" multiple accept=".mhd,.raw" className="hidden" onChange={(e) => handleFiles(e.target.files)} />
            <div className="text-sm font-medium text-ink">Click to upload the CT .mhd + .raw pair</div>
            <div className="text-xs text-ink-faint">Select both files together</div>
          </label>

          <div className="mt-3 flex gap-4 text-xs font-mono-tabular text-ink-muted">
            <span className={mhdFile ? 'text-positive' : ''}>{mhdFile ? '✓' : '○'} .mhd header</span>
            <span className={rawFile ? 'text-positive' : ''}>{rawFile ? '✓' : '○'} .raw data</span>
          </div>

          <button className="btn-primary mt-5" disabled={!bothFilesSelected || previewLoading} onClick={() => loadPreview()}>
            {previewLoading ? <Spinner /> : null}
            {previewLoading ? 'Loading preview…' : 'Load Slice Preview'}
          </button>
        </>
      )}

      {error && (
        <div className="mt-3 rounded-lg border border-danger/40 bg-danger-soft px-3 py-2 text-sm text-danger">
          {error}
        </div>
      )}

      {preview && (
        <div>
          <p className="text-sm text-ink-muted mb-3">
            Click a point on the scan to select the candidate location the 3D model will classify.
          </p>

          <div className="relative w-full max-w-xl rounded-lg overflow-hidden border border-border bg-black">
            <img
              ref={imgRef}
              src={`data:image/png;base64,${preview.image_base64}`}
              alt={`CT slice ${preview.slice_index}`}
              className="block w-full aspect-square cursor-crosshair"
              onClick={handleImageClick}
            />
            {picked && (
              <div
                className="absolute w-4 h-4 -ml-2 -mt-2 rounded-full border-2 border-accent shadow-[0_0_0_2px_rgba(0,0,0,0.6)]"
                style={{
                  left: `${(picked.col / preview.width) * 100}%`,
                  top: `${(picked.row / preview.height) * 100}%`,
                }}
              />
            )}
          </div>

          <div className="mt-3 flex items-center gap-3">
            <span className="text-xs text-ink-faint font-mono-tabular w-24">
              Slice {sliderValue + 1}/{preview.num_slices}
              {previewLoading && '…'}
            </span>
            <input
              type="range"
              min={0}
              max={preview.num_slices - 1}
              value={sliderValue}
              // Moves instantly, no request -- only fetches the slice once
              // the user actually settles on one (release, or arrow-key up).
              onChange={(e) => setSliderValue(Number(e.target.value))}
              onMouseUp={(e) => loadPreview(Number(e.currentTarget.value))}
              onTouchEnd={(e) => loadPreview(Number(e.currentTarget.value))}
              onKeyUp={(e) => loadPreview(Number(e.currentTarget.value))}
              className="flex-1 accent-accent"
            />
          </div>

          {picked && (
            <div className="mt-4 card !py-3 !px-4 flex items-center justify-between">
              <div className="text-xs font-mono-tabular text-ink-muted">
                world coord: ({picked.world.map((v) => v.toFixed(1)).join(', ')}) mm
              </div>
              <button
                className="btn-primary !py-2"
                disabled={loading}
                onClick={() => mhdFile && rawFile && onAnalyze(mhdFile, rawFile, picked.world)}
              >
                {loading ? <Spinner /> : null}
                {loading ? 'Analyzing…' : 'Analyze This Location'}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
