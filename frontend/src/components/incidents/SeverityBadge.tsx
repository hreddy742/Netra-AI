'use client'

import type { Severity } from '@/types'

interface SeverityBadgeProps {
  severity: Severity
  size?: 'sm' | 'md'
}

const SEVERITY_STYLES: Record<Severity, string> = {
  LOW: 'bg-gray-500/20 text-gray-400 border border-gray-500/40',
  MEDIUM: 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/40',
  HIGH: 'bg-orange-500/20 text-orange-400 border border-orange-500/40',
  CRITICAL: 'bg-red-500/20 text-red-400 border border-red-500/40',
}

export function SeverityBadge({ severity, size = 'sm' }: SeverityBadgeProps) {
  const sizeClass = size === 'sm' ? 'text-xs px-2 py-0.5' : 'text-sm px-3 py-1'
  return (
    <span
      className={`inline-flex items-center rounded font-semibold uppercase tracking-wide ${sizeClass} ${SEVERITY_STYLES[severity]}`}
    >
      {severity}
    </span>
  )
}
