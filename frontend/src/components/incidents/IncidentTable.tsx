'use client'

import { useRouter } from 'next/navigation'
import type { Incident } from '@/types'
import { SeverityBadge } from './SeverityBadge'
import { StatusBadge } from '@/components/common/StatusBadge'
import { RiskBar } from '@/components/common/RiskBar'

interface IncidentTableProps {
  incidents: Incident[]
  loading: boolean
  total: number
  page: number
  pageSize: number
  onPageChange: (page: number) => void
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

export function IncidentTable({
  incidents,
  loading,
  total,
  page,
  pageSize,
  onPageChange,
}: IncidentTableProps) {
  const router = useRouter()
  const totalPages = Math.ceil(total / pageSize)

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
        <div className="p-8 flex items-center justify-center">
          <div className="flex items-center gap-3 text-gray-400">
            <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            <span>Loading incidents...</span>
          </div>
        </div>
      </div>
    )
  }

  if (incidents.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-12 text-center">
        <svg className="w-10 h-10 text-gray-600 mx-auto mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 002.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 00-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75 2.25 2.25 0 00-.1-.664m-5.8 0A2.251 2.251 0 0113.5 2.25H15c1.012 0 1.867.668 2.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m0 0H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V9.375c0-.621-.504-1.125-1.125-1.125H8.25z" />
        </svg>
        <p className="text-gray-400">No incidents found</p>
        <p className="text-gray-600 text-sm mt-1">Try adjusting your filters</p>
      </div>
    )
  }

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-700 text-gray-400 text-xs uppercase tracking-wider">
            <th className="text-left px-4 py-3 font-medium">Time</th>
            <th className="text-left px-4 py-3 font-medium">Camera</th>
            <th className="text-left px-4 py-3 font-medium">Stage</th>
            <th className="text-left px-4 py-3 font-medium">Concealment</th>
            <th className="text-left px-4 py-3 font-medium w-32">Risk</th>
            <th className="text-left px-4 py-3 font-medium">Severity</th>
            <th className="text-left px-4 py-3 font-medium">Status</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-700/50">
          {incidents.map((inc) => (
            <tr
              key={inc.id}
              onClick={() => router.push(`/incidents/${inc.id}`)}
              className="hover:bg-gray-700/40 cursor-pointer transition-colors"
            >
              <td className="px-4 py-3 text-gray-300 whitespace-nowrap">
                {formatDate(inc.occurred_at)}
              </td>
              <td className="px-4 py-3 text-gray-300 font-mono text-xs">
                {inc.camera_id}
              </td>
              <td className="px-4 py-3">
                <span className="text-gray-300 text-xs font-mono">{inc.theft_stage}</span>
              </td>
              <td className="px-4 py-3 text-gray-400 text-xs">
                {inc.concealment_type || '—'}
              </td>
              <td className="px-4 py-3">
                <div className="flex items-center gap-2">
                  <div className="flex-1">
                    <RiskBar score={inc.risk_score} />
                  </div>
                  <span className="text-xs text-gray-400 w-8 text-right flex-shrink-0">
                    {(inc.risk_score * 100).toFixed(0)}%
                  </span>
                </div>
              </td>
              <td className="px-4 py-3">
                <SeverityBadge severity={inc.severity} />
              </td>
              <td className="px-4 py-3">
                <StatusBadge status={inc.status} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="border-t border-gray-700 px-4 py-3 flex items-center justify-between text-sm">
          <span className="text-gray-500">
            Showing {page * pageSize + 1}–{Math.min((page + 1) * pageSize, total)} of {total}
          </span>
          <div className="flex items-center gap-1">
            <button
              onClick={() => onPageChange(page - 1)}
              disabled={page === 0}
              className="px-2.5 py-1 rounded bg-gray-700 text-gray-300 hover:bg-gray-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              Prev
            </button>
            {Array.from({ length: Math.min(totalPages, 7) }).map((_, i) => {
              const p = i
              return (
                <button
                  key={p}
                  onClick={() => onPageChange(p)}
                  className={`px-2.5 py-1 rounded transition-colors ${
                    p === page
                      ? 'bg-red-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  {p + 1}
                </button>
              )
            })}
            <button
              onClick={() => onPageChange(page + 1)}
              disabled={page >= totalPages - 1}
              className="px-2.5 py-1 rounded bg-gray-700 text-gray-300 hover:bg-gray-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
