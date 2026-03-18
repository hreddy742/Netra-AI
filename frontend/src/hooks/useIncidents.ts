'use client'

import { useEffect } from 'react'
import { useIncidentsStore } from '@/stores/incidents'
import { useSSE } from './useSSE'
import type { IncidentSSEEvent } from '@/types'
import { getIncidentStreamUrl } from '@/lib/api'

export function useIncidents() {
  const store = useIncidentsStore()

  useEffect(() => {
    store.fetchIncidents()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [store.filters, store.page])

  useSSE<IncidentSSEEvent>({
    url: getIncidentStreamUrl(),
    onMessage: (event) => {
      if (event.type === 'new_incident' || event.type === 'incident_updated') {
        store.upsertIncident(event.incident)
      }
    },
  })

  return {
    incidents: store.incidents,
    total: store.total,
    loading: store.loading,
    error: store.error,
    filters: store.filters,
    page: store.page,
    pageSize: store.pageSize,
    setPage: store.setPage,
    setSeverityFilter: store.setSeverityFilter,
    setStatusFilter: store.setStatusFilter,
    setCameraFilter: store.setCameraFilter,
    refetch: store.fetchIncidents,
  }
}
