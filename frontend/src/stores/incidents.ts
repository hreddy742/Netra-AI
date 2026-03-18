'use client'

import { create } from 'zustand'
import type { Incident, IncidentFilters, Severity, IncidentStatus } from '@/types'
import { getIncidents } from '@/lib/api'

interface IncidentsState {
  incidents: Incident[]
  total: number
  loading: boolean
  error: string | null
  filters: IncidentFilters
  page: number
  pageSize: number

  setFilters: (filters: Partial<IncidentFilters>) => void
  setPage: (page: number) => void
  fetchIncidents: () => Promise<void>
  upsertIncident: (incident: Incident) => void
  setSeverityFilter: (severity: Severity | undefined) => void
  setStatusFilter: (status: IncidentStatus | undefined) => void
  setCameraFilter: (camera_id: string | undefined) => void
}

export const useIncidentsStore = create<IncidentsState>((set, get) => ({
  incidents: [],
  total: 0,
  loading: false,
  error: null,
  filters: {},
  page: 0,
  pageSize: 20,

  setFilters: (filters) => {
    set((state) => ({ filters: { ...state.filters, ...filters }, page: 0 }))
  },

  setPage: (page) => set({ page }),

  setSeverityFilter: (severity) => {
    set((state) => ({
      filters: { ...state.filters, severity },
      page: 0,
    }))
  },

  setStatusFilter: (status) => {
    set((state) => ({
      filters: { ...state.filters, status },
      page: 0,
    }))
  },

  setCameraFilter: (camera_id) => {
    set((state) => ({
      filters: { ...state.filters, camera_id },
      page: 0,
    }))
  },

  fetchIncidents: async () => {
    const { filters, page, pageSize } = get()
    set({ loading: true, error: null })
    try {
      const result = await getIncidents({
        ...filters,
        limit: pageSize,
        offset: page * pageSize,
      })
      set({ incidents: result.incidents, total: result.total, loading: false })
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : 'Failed to fetch incidents',
        loading: false,
      })
    }
  },

  upsertIncident: (incident) => {
    set((state) => {
      const idx = state.incidents.findIndex((i) => i.id === incident.id)
      if (idx >= 0) {
        const updated = [...state.incidents]
        updated[idx] = incident
        return { incidents: updated }
      }
      return {
        incidents: [incident, ...state.incidents],
        total: state.total + 1,
      }
    })
  },
}))
