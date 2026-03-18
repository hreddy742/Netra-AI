'use client'

import { useEffect, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import type { Incident } from '@/types'
import { getIncident } from '@/lib/api'
import { SeverityBadge } from '@/components/incidents/SeverityBadge'
import { StatusBadge } from '@/components/common/StatusBadge'
import { ReviewPanel } from '@/components/incidents/ReviewPanel'
import { RiskBar } from '@/components/common/RiskBar'

function DetailRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-start gap-4 py-3 border-b border-gray-700/50 last:border-0">
      <span className="text-xs text-gray-500 uppercase tracking-wide font-medium w-40 flex-shrink-0 pt-0.5">
        {label}
      </span>
      <span className="text-sm text-gray-200">{value}</span>
    </div>
  )
}

export default function IncidentDetailPage() {
  const { id } = useParams<{ id: string }>()
  const router = useRouter()
  const [incident, setIncident] = useState<Incident | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    getIncident(id)
      .then((data) => {
        if (!cancelled) {
          setIncident(data)
          setLoading(false)
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load incident')
          setLoading(false)
        }
      })
    return () => {
      cancelled = true
    }
  }, [id])

  if (loading) {
    return (
      <div className="p-6 flex items-center justify-center py-20">
        <div className="flex items-center gap-3 text-gray-400">
          <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          Loading incident...
        </div>
      </div>
    )
  }

  if (error || !incident) {
    return (
      <div className="p-6">
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <div className="w-12 h-12 rounded-full bg-red-600/20 flex items-center justify-center mb-3">
            <svg className="w-6 h-6 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.072 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
          </div>
          <p className="text-red-400 font-medium">{error || 'Incident not found'}</p>
          <button
            onClick={() => router.back()}
            className="mt-4 px-4 py-2 bg-gray-800 border border-gray-700 text-gray-300 rounded text-sm hover:bg-gray-700 transition-colors"
          >
            Go back
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 max-w-4xl space-y-6">
      {/* Breadcrumb + header */}
      <div>
        <button
          onClick={() => router.push('/incidents')}
          className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-300 mb-3 transition-colors"
        >
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Back to Incidents
        </button>
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-xl font-bold text-gray-100">Incident Detail</h1>
            <p className="text-sm text-gray-500 mt-0.5 font-mono">{incident.id}</p>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            <SeverityBadge severity={incident.severity} size="md" />
            <StatusBadge status={incident.status} size="md" />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: incident details */}
        <div className="lg:col-span-2 space-y-4">
          {/* Risk score card */}
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-3">
            <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wide">Risk Assessment</h2>
            <div className="space-y-3">
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Overall Risk Score</span>
                  <span className="text-gray-200 font-semibold">
                    {(incident.risk_score * 100).toFixed(1)}%
                  </span>
                </div>
                <RiskBar score={incident.risk_score} height="h-3" />
              </div>

              <div className="grid grid-cols-2 gap-3 pt-1">
                <div className="bg-gray-900/50 rounded p-3 space-y-1">
                  <p className="text-xs text-gray-500">FSM Score</p>
                  <p className="text-lg font-bold text-gray-200">
                    {incident.fsm_score.toFixed(1)}
                  </p>
                  <RiskBar score={incident.fsm_score / 10} />
                </div>
                <div className="bg-gray-900/50 rounded p-3 space-y-1">
                  <p className="text-xs text-gray-500">Shopformer Score</p>
                  <p className="text-lg font-bold text-gray-200">
                    {(incident.shopformer_score * 100).toFixed(1)}%
                  </p>
                  <RiskBar score={incident.shopformer_score} />
                </div>
              </div>
            </div>
          </div>

          {/* Detail table */}
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
            <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-2">
              Incident Details
            </h2>
            <div>
              <DetailRow label="Camera ID" value={<span className="font-mono">{incident.camera_id}</span>} />
              <DetailRow label="Track ID" value={`#${incident.track_id}`} />
              <DetailRow
                label="Theft Stage"
                value={
                  <span className="font-mono text-xs bg-gray-700 px-2 py-0.5 rounded">
                    {incident.theft_stage}
                  </span>
                }
              />
              <DetailRow
                label="Concealment"
                value={incident.concealment_type || <span className="text-gray-500 italic">None detected</span>}
              />
              <DetailRow label="Severity" value={<SeverityBadge severity={incident.severity} />} />
              <DetailRow label="Status" value={<StatusBadge status={incident.status} />} />
              <DetailRow
                label="Occurred At"
                value={new Date(incident.occurred_at).toLocaleString()}
              />
              {incident.operator_notes && (
                <DetailRow
                  label="Operator Notes"
                  value={
                    <span className="text-gray-300 whitespace-pre-wrap">{incident.operator_notes}</span>
                  }
                />
              )}
            </div>
          </div>
        </div>

        {/* Right: review panel */}
        <div className="lg:col-span-1">
          <ReviewPanel
            incident={incident}
            onReviewed={(updated) => setIncident(updated)}
          />
        </div>
      </div>
    </div>
  )
}
