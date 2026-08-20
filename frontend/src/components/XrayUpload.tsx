import { useState } from 'react'

interface Props {
  onAnalyze: (file: File) => void
  loading: boolean
}

export default function XrayUpload({ onAnalyze, loading }: Props) {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)

  function handleFile(f: File | null) {
    setFile(f)
    setPreviewUrl(f ? URL.createObjectURL(f) : null)
  }

  return (
    <div className="max-w-xl">
      <label
        className="flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed
          border-border bg-surface-2 px-6 py-10 text-center cursor-pointer transition-colors
          hover:border-accent/60"
      >
        <input
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
        />
        {previewUrl ? (
          <img src={previewUrl} alt="X-ray preview" className="max-h-56 rounded-lg" />
        ) : (
          <>
            <div className="text-sm font-medium text-ink">Click to upload a chest X-ray</div>
            <div className="text-xs text-ink-faint">JPEG or PNG</div>
          </>
        )}
      </label>

      {file && (
        <div className="mt-3 flex items-center justify-between text-xs text-ink-muted font-mono-tabular">
          <span>{file.name}</span>
          <span>{(file.size / 1024).toFixed(0)} KB</span>
        </div>
      )}

      <button
        className="btn-primary mt-5"
        disabled={!file || loading}
        onClick={() => file && onAnalyze(file)}
      >
        {loading ? <Spinner /> : null}
        {loading ? 'Analyzing…' : 'Analyze X-ray'}
      </button>
    </div>
  )
}

export function Spinner() {
  return (
    <svg className="animate-spin w-3.5 h-3.5" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-90" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
    </svg>
  )
}
