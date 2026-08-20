// A custom in-app confirmation modal (not the native window.confirm())
// for irreversible actions -- Day 12 delete-report feature. Native
// confirm() is jarring/inconsistent with the rest of the design system
// and can't be styled, so every destructive action in this app routes
// through this instead.

interface Props {
  open: boolean
  title: string
  description: string
  confirmLabel?: string
  danger?: boolean
  busy?: boolean
  onConfirm: () => void
  onCancel: () => void
}

export default function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel = 'Confirm',
  danger = true,
  busy = false,
  onConfirm,
  onCancel,
}: Props) {
  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 px-4"
      onClick={onCancel}
    >
      <div
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-dialog-title"
        className="card w-full max-w-sm"
        onClick={(e) => e.stopPropagation()}
      >
        <div id="confirm-dialog-title" className="text-base font-semibold text-ink mb-1.5">
          {title}
        </div>
        <p className="text-sm text-ink-muted leading-relaxed mb-5">{description}</p>
        <div className="flex justify-end gap-2">
          <button type="button" className="btn-secondary" onClick={onCancel} disabled={busy}>
            Cancel
          </button>
          <button
            type="button"
            className={danger ? 'btn-primary !bg-danger !text-white' : 'btn-primary'}
            onClick={onConfirm}
            disabled={busy}
          >
            {busy ? 'Deleting…' : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}
