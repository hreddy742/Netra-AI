'use client'

import { useHealth } from '@/hooks/useHealth'

export function TopBar() {
  const { status, error } = useHealth(10000)

  const isHealthy = !error && status?.service === 'ok'
  const isUnknown = !error && status === null

  return (
    <header className="h-12 bg-gray-900 border-b border-gray-800 flex items-center justify-between px-6 flex-shrink-0">
      <div className="text-gray-400 text-sm font-medium">
        Security Operations Center
      </div>

      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${
              isUnknown
                ? 'bg-gray-500'
                : isHealthy
                ? 'bg-green-500 animate-pulse'
                : 'bg-red-500 animate-pulse'
            }`}
          />
          <span className="text-xs text-gray-400">
            {isUnknown ? 'Connecting...' : isHealthy ? 'All systems operational' : 'System degraded'}
          </span>
        </div>

        <div className="w-px h-4 bg-gray-700" />

        <span className="text-xs text-gray-500">
          {new Date().toLocaleTimeString()}
        </span>
      </div>
    </header>
  )
}
