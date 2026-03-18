'use client'

import { useIncidents } from '@/hooks/useIncidents'
import { useCameras } from '@/hooks/useCameras'
import { IncidentFilters } from '@/components/incidents/IncidentFilters'
import { IncidentTable } from '@/components/incidents/IncidentTable'

export default function IncidentsPage() {
  const {
    incidents,
    total,
    loading,
    error,
    filters,
    page,
    pageSize,
    setPage,
    setSeverityFilter,
    setStatusFilter,
    setCameraFilter,
    refetch,
  } = useIncidents()

  const { cameras } = useCameras()

  return (
    <div className="p-6 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-gray-100">Incidents</h1>
          <p className="text-sm text-gray-500 mt-0.5">
            {loading ? 'Loading...' : `${total} total incidents`}
          </p>
        </div>
        <button
          onClick={refetch}
          className="flex items-center gap-2 px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm text-gray-300 hover:bg-gray-700 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Refresh
        </button>
      </div>

      {/* Filters */}
      <IncidentFilters
        cameras={cameras}
        severity={filters.severity}
        status={filters.status}
        cameraId={filters.camera_id}
        onSeverityChange={setSeverityFilter}
        onStatusChange={setStatusFilter}
        onCameraChange={setCameraFilter}
      />

      {/* Error */}
      {error && (
        <div className="flex items-center gap-3 bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3 text-red-400 text-sm">
          <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.072 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
          {error}
        </div>
      )}

      {/* Table */}
      <IncidentTable
        incidents={incidents}
        loading={loading}
        total={total}
        page={page}
        pageSize={pageSize}
        onPageChange={setPage}
      />
    </div>
  )
}
