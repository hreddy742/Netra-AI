'use client'

import type { IncidentStatus } from '@/types'

interface StatusBadgeProps {
  status: IncidentStatus
  size?: 'sm' | 'md'
}

const STATUS_STYLES: Record<IncidentStatus, string> = {
  OPEN: 'bg-blue-500/20 text-blue-400 border border-blue-500/40',
  REVIEWING: 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/40',
  CONFIRMED: 'bg-red-500/20 text-red-400 border border-red-500/40',
  DISMISSED: 'bg-gray-500/20 text-gray-400 border border-gray-500/40',
}

export function StatusBadge({ status, size = 'sm' }: StatusBadgeProps) {
  const sizeClass = size === 'sm' ? 'text-xs px-2 py-0.5' : 'text-sm px-3 py-1'
  return (
    <span className={`inline-flex items-center rounded font-medium ${sizeClass} ${STATUS_STYLES[status]}`}>
      {status}
    </span>
  )
}
