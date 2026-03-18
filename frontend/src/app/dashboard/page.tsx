'use client'

import { useEffect } from 'react'
import { CameraGrid } from '@/components/dashboard/CameraGrid'
import { useIncidentsStore } from '@/stores/incidents'
import { useSSE } from '@/hooks/useSSE'
import type { IncidentSSEEvent } from '@/types'
import { getIncidentStreamUrl } from '@/lib/api'

function OpenIncidentBadge() {
  const incidents = useIncidentsStore((s) => s.incidents)
  const fetchIncidents = useIncidentsStore((s) => s.fetchIncidents)
  const upsertIncident = useIncidentsStore((s) => s.upsertIncident)

  const openCount = incidents.filter((i) => i.status === 'OPEN').length
  const criticalCount = incidents.filter((i) => i.severity === 'CRITICAL' && i.status === 'OPEN').length

  // Fetch open incidents for the badge count
  useEffect(() => {
    fetchIncidents()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Subscribe to SSE for real-time incident updates
  useSSE<IncidentSSEEvent>({
    url: getIncidentStreamUrl(),
    onMessage: (event) => {
      if (event.type === 'new_incident' || event.type === 'incident_updated') {
        upsertIncident(event.incident)
      }
    },
  })

  return (
    <div className="flex items-center gap-4">
      <div className="flex items-center gap-2 bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5">
        <span className="text-xs text-gray-500 uppercase tracking-wide font-medium">Open Incidents</span>
        <span className={`text-lg font-bold ${openCount > 0 ? 'text-red-400' : 'text-green-400'}`}>
          {openCount}
        </span>
      </div>
      {criticalCount > 0 && (
        <div className="flex items-center gap-2 bg-red-600/10 border border-red-600/30 rounded-lg px-4 py-2.5 animate-pulse">
          <div className="w-2 h-2 rounded-full bg-red-500" />
          <span className="text-xs text-red-400 font-semibold uppercase tracking-wide">
            {criticalCount} Critical
          </span>
        </div>
      )}
    </div>
  )
}

export default function DashboardPage() {
  return (
    <div className="p-6 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-gray-100">Live Dashboard</h1>
          <p className="text-sm text-gray-500 mt-0.5">Real-time camera feeds and risk monitoring</p>
        </div>
        <OpenIncidentBadge />
      </div>

      {/* Camera grid */}
      <CameraGrid />
    </div>
  )
}
