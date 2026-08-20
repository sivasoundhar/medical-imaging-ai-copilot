import { useCallback, useRef, useState } from 'react'
import { CompareArrowsIcon } from './icons'

// Drag-to-compare (mockup's "Study Overview" handle) over the exact same
// underlying image pair AnalysisResult already used side-by-side --
// resized_original_base64 and heatmap_base64 are guaranteed
// pixel-dimension-identical by construction (see AnalysisResult.tsx's
// comment on that), so no letterboxing/cropping mismatch here either.

interface Props {
  originalSrc: string
  overlaySrc: string
}

export default function CompareSlider({ originalSrc, overlaySrc }: Props) {
  const [percent, setPercent] = useState(50)
  const containerRef = useRef<HTMLDivElement>(null)
  const draggingRef = useRef(false)

  const updateFromClientX = useCallback((clientX: number) => {
    const el = containerRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    const pct = ((clientX - rect.left) / rect.width) * 100
    setPercent(Math.min(100, Math.max(0, pct)))
  }, [])

  const onPointerDown = (e: React.PointerEvent) => {
    draggingRef.current = true
    ;(e.target as Element).setPointerCapture(e.pointerId)
    updateFromClientX(e.clientX)
  }
  const onPointerMove = (e: React.PointerEvent) => {
    if (!draggingRef.current) return
    updateFromClientX(e.clientX)
  }
  const onPointerUp = () => {
    draggingRef.current = false
  }
  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowLeft') setPercent((p) => Math.max(0, p - 5))
    if (e.key === 'ArrowRight') setPercent((p) => Math.min(100, p + 5))
  }

  return (
    <div
      ref={containerRef}
      className="relative aspect-square w-full rounded-lg border border-border bg-black/20 overflow-hidden select-none touch-none"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    >
      {/* Bottom layer: Grad-CAM overlay, always full-frame */}
      <img
        src={overlaySrc}
        alt="Grad-CAM heatmap overlay"
        className="absolute inset-0 w-full h-full object-contain pointer-events-none"
        draggable={false}
      />
      {/* Top layer: original, clipped to the left `percent` of the frame */}
      <div
        className="absolute inset-0 overflow-hidden pointer-events-none"
        style={{ clipPath: `inset(0 ${100 - percent}% 0 0)` }}
      >
        <img
          src={originalSrc}
          alt="Original X-ray"
          className="absolute inset-0 w-full h-full object-contain"
          draggable={false}
        />
      </div>

      {/* Divider + drag handle */}
      <div
        className="absolute inset-y-0 w-0.5 bg-white/80 pointer-events-none"
        style={{ left: `${percent}%` }}
      />
      <button
        type="button"
        aria-label="Drag to compare original and Grad-CAM overlay"
        className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-9 h-9 rounded-full bg-white text-ink-faint
          border border-border shadow-lg flex items-center justify-center cursor-ew-resize outline-none
          focus-visible:ring-2 focus-visible:ring-accent"
        style={{ left: `${percent}%` }}
        onKeyDown={onKeyDown}
      >
        <CompareArrowsIcon className="w-4.5 h-4.5" />
      </button>

      <span className="absolute top-2 left-2 rounded bg-black/50 px-1.5 py-0.5 text-[10px] text-white pointer-events-none">
        Original
      </span>
      <span className="absolute top-2 right-2 rounded bg-black/50 px-1.5 py-0.5 text-[10px] text-white pointer-events-none">
        Grad-CAM
      </span>
    </div>
  )
}
