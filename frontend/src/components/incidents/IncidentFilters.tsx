'use client'

import type { Camera, Severity, IncidentStatus } from '@/types'

interface IncidentFiltersProps {
  cameras: Camera[]
  severity: Severity | undefined
  status: IncidentStatus | undefined
  cameraId: string | undefined
  onSeverityChange: (v: Severity | undefined) => void
  onStatusChange: (v: IncidentStatus | undefined) => void
  onCameraChange: (v: string | undefined) => void
}

const SEVERITIES: Severity[] = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
const STATUSES: IncidentStatus[] = ['OPEN', 'REVIEWING', 'CONFIRMED', 'DISMISSED']

const SELECT_CLS =
  'bg-gray-800 border border-gray-700 text-gray-200 text-sm rounded px-3 py-1.5 focus:outline-none focus:border-gray-500 hover:border-gray-600 transition-colors'

export function IncidentFilters({
  cameras,
  severity,
  status,
  cameraId,
  onSeverityChange,
  onStatusChange,
  onCameraChange,
}: IncidentFiltersProps) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <span className="text-sm text-gray-500 font-medium">Filter:</span>

      {/* Severity */}
      <select
        value={severity ?? ''}
        onChange={(e) => onSeverityChange((e.target.value as Severity) || undefined)}
        className={SELECT_CLS}
      >
        <option value="">All Severities</option>
        {SEVERITIES.map((s) => (
          <option key={s} value={s}>
            {s}
          </option>
        ))}
      </select>

      {/* Status */}
      <select
        value={status ?? ''}
        onChange={(e) => onStatusChange((e.target.value as IncidentStatus) || undefined)}
        className={SELECT_CLS}
      >
        <option value="">All Statuses</option>
        {STATUSES.map((s) => (
          <option key={s} value={s}>
            {s}
          </option>
        ))}
      </select>

      {/* Camera */}
      <select
        value={cameraId ?? ''}
        onChange={(e) => onCameraChange(e.target.value || undefined)}
        className={SELECT_CLS}
      >
        <option value="">All Cameras</option>
        {cameras.map((c) => (
          <option key={c.id} value={c.id}>
            {c.display_name}
          </option>
        ))}
      </select>

      {/* Clear */}
      {(severity || status || cameraId) && (
        <button
          onClick={() => {
            onSeverityChange(undefined)
            onStatusChange(undefined)
            onCameraChange(undefined)
          }}
          className="text-xs text-gray-500 hover:text-gray-300 underline transition-colors"
        >
          Clear filters
        </button>
      )}
    </div>
  )
}
