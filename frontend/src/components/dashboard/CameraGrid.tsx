'use client'

import { useEffect } from 'react'
import { useIncidentsStore } from '@/stores/incidents'
import { useCameras } from '@/hooks/useCameras'
import { CameraCard } from './CameraCard'

export function CameraGrid() {
  const { cameras, loading, error, refetch } = useCameras()
  const incidents = useIncidentsStore((s) => s.incidents)

  // Count open incidents per camera
  const openCountByCam = incidents.reduce<Record<string, number>>((acc, inc) => {
    if (inc.status === 'OPEN') {
      acc[inc.camera_id] = (acc[inc.camera_id] ?? 0) + 1
    }
    return acc
  }, {})

  useEffect(() => {
    refetch()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  if (loading) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="bg-gray-800 rounded-lg border border-gray-700 aspect-video animate-pulse" />
        ))}
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-center">
        <div className="w-12 h-12 rounded-full bg-red-600/20 flex items-center justify-center mb-3">
          <svg className="w-6 h-6 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.072 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
        </div>
        <p className="text-red-400 font-medium">Failed to load cameras</p>
        <p className="text-gray-500 text-sm mt-1">{error}</p>
        <button
          onClick={refetch}
          className="mt-4 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 rounded text-sm transition-colors"
        >
          Retry
        </button>
      </div>
    )
  }

  if (cameras.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-center">
        <svg className="w-12 h-12 text-gray-600 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
        </svg>
        <p className="text-gray-400 font-medium">No cameras configured</p>
        <p className="text-gray-600 text-sm mt-1">Add cameras in the Cameras section</p>
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
      {cameras.map((camera) => (
        <CameraCard
          key={camera.id}
          camera={camera}
          openIncidentCount={openCountByCam[camera.id] ?? 0}
        />
      ))}
    </div>
  )
}
