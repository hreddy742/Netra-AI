'use client'

import { useState } from 'react'
import type { Camera } from '@/types'
import { getMjpegStreamUrl } from '@/lib/api'
import { useCameraRiskScore } from '@/hooks/useCameras'
import { RiskBar } from '@/components/common/RiskBar'

interface CameraCardProps {
  camera: Camera
  openIncidentCount?: number
}

function getRiskBorderColor(score: number): string {
  if (score < 0.33) return 'border-green-600/40'
  if (score < 0.66) return 'border-yellow-500/40'
  if (score < 0.85) return 'border-orange-500/40'
  return 'border-red-500/50'
}

function getRiskHeaderColor(score: number): string {
  if (score < 0.33) return 'text-green-400'
  if (score < 0.66) return 'text-yellow-400'
  if (score < 0.85) return 'text-orange-400'
  return 'text-red-400'
}

export function CameraCard({ camera, openIncidentCount = 0 }: CameraCardProps) {
  const riskScore = useCameraRiskScore(camera.id)
  const [imgError, setImgError] = useState(false)
  const borderColor = getRiskBorderColor(riskScore)
  const scoreColor = getRiskHeaderColor(riskScore)

  return (
    <div
      className={`bg-gray-800 rounded-lg border ${borderColor} overflow-hidden flex flex-col transition-all duration-300`}
    >
      {/* Stream */}
      <div className="relative aspect-video bg-gray-900 overflow-hidden">
        {camera.enabled && !imgError ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={getMjpegStreamUrl(camera.id)}
            className="w-full h-full object-cover"
            alt={`Stream: ${camera.display_name}`}
            onError={() => setImgError(true)}
          />
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center text-gray-600">
            <svg className="w-10 h-10 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
            </svg>
            <span className="text-xs">{camera.enabled ? 'Stream unavailable' : 'Camera disabled'}</span>
          </div>
        )}

        {/* Overlay badges */}
        <div className="absolute top-2 left-2 flex gap-1.5">
          <span className={`text-xs font-bold px-1.5 py-0.5 rounded bg-black/60 ${scoreColor}`}>
            {(riskScore * 100).toFixed(0)}%
          </span>
        </div>

        {openIncidentCount > 0 && (
          <div className="absolute top-2 right-2">
            <span className="text-xs font-bold px-1.5 py-0.5 rounded bg-red-600 text-white">
              {openIncidentCount} OPEN
            </span>
          </div>
        )}

        {/* Live dot */}
        {camera.enabled && !imgError && (
          <div className="absolute bottom-2 left-2 flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
            <span className="text-xs text-gray-300 bg-black/50 px-1 rounded">LIVE</span>
          </div>
        )}
      </div>

      {/* Info bar */}
      <div className="px-3 pt-2.5 pb-3 space-y-2">
        <div className="flex items-center justify-between">
          <div className="min-w-0">
            <p className="text-sm font-semibold text-gray-100 truncate">{camera.display_name}</p>
            <p className="text-xs text-gray-500 truncate">{camera.location}</p>
          </div>
          <span
            className={`text-xs px-1.5 py-0.5 rounded border flex-shrink-0 ml-2 ${
              camera.enabled
                ? 'text-green-400 border-green-600/40 bg-green-600/10'
                : 'text-gray-500 border-gray-600/40 bg-gray-700/30'
            }`}
          >
            {camera.enabled ? 'ON' : 'OFF'}
          </span>
        </div>

        <div className="space-y-1">
          <div className="flex justify-between text-xs text-gray-500">
            <span>Risk Score</span>
            <span className={scoreColor}>{(riskScore * 100).toFixed(0)}%</span>
          </div>
          <RiskBar score={riskScore} />
        </div>
      </div>
    </div>
  )
}
