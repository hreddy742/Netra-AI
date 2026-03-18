'use client'

import { create } from 'zustand'
import type { Camera, CameraFormData } from '@/types'
import { getCameras, createCamera } from '@/lib/api'

interface RiskScore {
  score: number
  updatedAt: number
}

interface CamerasState {
  cameras: Camera[]
  loading: boolean
  error: string | null
  riskScores: Record<string, RiskScore>

  fetchCameras: () => Promise<void>
  addCamera: (data: CameraFormData) => Promise<void>
  setRiskScore: (cameraId: string, score: number) => void
  getRiskScore: (cameraId: string) => number
}

export const useCamerasStore = create<CamerasState>((set, get) => ({
  cameras: [],
  loading: false,
  error: null,
  riskScores: {},

  fetchCameras: async () => {
    set({ loading: true, error: null })
    try {
      const cameras = await getCameras()
      set({ cameras, loading: false })
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : 'Failed to fetch cameras',
        loading: false,
      })
    }
  },

  addCamera: async (data: CameraFormData) => {
    set({ loading: true, error: null })
    try {
      const camera = await createCamera(data)
      set((state) => ({
        cameras: [...state.cameras, camera],
        loading: false,
      }))
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : 'Failed to create camera',
        loading: false,
      })
      throw err
    }
  },

  setRiskScore: (cameraId, score) => {
    set((state) => ({
      riskScores: {
        ...state.riskScores,
        [cameraId]: { score, updatedAt: Date.now() },
      },
    }))
  },

  getRiskScore: (cameraId) => {
    const entry = get().riskScores[cameraId]
    return entry?.score ?? 0
  },
}))
