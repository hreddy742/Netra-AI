'use client'

import { useEffect } from 'react'
import { useCamerasStore } from '@/stores/cameras'
import { useSSE } from './useSSE'
import type { RiskScoreEvent } from '@/types'
import { getRiskStreamUrl } from '@/lib/api'

export function useCameras() {
  const store = useCamerasStore()

  useEffect(() => {
    store.fetchCameras()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return {
    cameras: store.cameras,
    loading: store.loading,
    error: store.error,
    addCamera: store.addCamera,
    refetch: store.fetchCameras,
    riskScores: store.riskScores,
  }
}

// Hook for subscribing to risk score updates for a single camera
export function useCameraRiskScore(cameraId: string) {
  const setRiskScore = useCamerasStore((s) => s.setRiskScore)
  const riskScore = useCamerasStore((s) => s.riskScores[cameraId]?.score ?? 0)

  useSSE<RiskScoreEvent>({
    url: getRiskStreamUrl(cameraId),
    onMessage: (event) => {
      setRiskScore(event.camera_id, event.risk_score)
    },
  })

  return riskScore
}
