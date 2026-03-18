'use client'

import { useHealth } from '@/hooks/useHealth'

interface ServiceCardProps {
  name: string
  status: 'ok' | 'down' | 'unknown'
  detail?: string
}

function ServiceCard({ name, status, detail }: ServiceCardProps) {
  const isOk = status === 'ok'
  const isUnknown = status === 'unknown'

  return (
    <div
      className={`bg-gray-800 rounded-lg border p-4 flex items-center gap-4 ${
        isUnknown
          ? 'border-gray-700'
          : isOk
          ? 'border-green-600/30'
          : 'border-red-600/30'
      }`}
    >
      {/* Status indicator */}
      <div
        className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 ${
          isUnknown
            ? 'bg-gray-700'
            : isOk
            ? 'bg-green-600/20'
            : 'bg-red-600/20'
        }`}
      >
        {isUnknown ? (
          <svg className="w-5 h-5 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        ) : isOk ? (
          <svg className="w-5 h-5 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        ) : (
          <svg className="w-5 h-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        )}
      </div>

      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold text-gray-200">{name}</p>
        {detail && <p className="text-xs text-gray-500 mt-0.5 truncate">{detail}</p>}
      </div>

      <span
        className={`text-xs font-bold px-2.5 py-1 rounded border flex-shrink-0 ${
          isUnknown
            ? 'text-gray-500 border-gray-600/40 bg-gray-700/30'
            : isOk
            ? 'text-green-400 border-green-600/40 bg-green-600/10'
            : 'text-red-400 border-red-600/40 bg-red-600/10'
        }`}
      >
        {isUnknown ? 'CHECKING' : isOk ? 'OPERATIONAL' : 'DOWN'}
      </span>
    </div>
  )
}

// Parse health response into named service entries
function parseHealthServices(
  status: Record<string, string> | null
): Array<{ name: string; status: 'ok' | 'down' | 'unknown'; detail?: string }> {
  if (!status) {
    return [
      { name: 'API Gateway', status: 'unknown' },
      { name: 'Detection Pipeline', status: 'unknown' },
      { name: 'Database', status: 'unknown' },
      { name: 'Stream Server', status: 'unknown' },
    ]
  }

  // Map known keys to display names
  const KEY_NAMES: Record<string, string> = {
    service: 'API Gateway',
    pipeline: 'Detection Pipeline',
    database: 'Database',
    stream: 'Stream Server',
    redis: 'Redis Cache',
    model: 'ML Model',
  }

  const entries = Object.entries(status)
  if (entries.length === 0) return []

  return entries.map(([key, value]) => ({
    name: KEY_NAMES[key] ?? key.charAt(0).toUpperCase() + key.slice(1),
    status: value === 'ok' ? 'ok' : 'down',
    detail: value !== 'ok' && value !== 'down' ? value : undefined,
  }))
}

export default function HealthPage() {
  const { status, loading, error, lastChecked, refresh } = useHealth(5000)

  const services = parseHealthServices(status as Record<string, string> | null)
  const allOk = !error && services.every((s) => s.status === 'ok')
  const anyDown = error || services.some((s) => s.status === 'down')

  return (
    <div className="p-6 max-w-2xl space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-gray-100">System Health</h1>
          <p className="text-sm text-gray-500 mt-0.5">
            Auto-refreshes every 5 seconds
            {lastChecked && ` · Last checked ${lastChecked.toLocaleTimeString()}`}
          </p>
        </div>
        <button
          onClick={refresh}
          className="flex items-center gap-2 px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm text-gray-300 hover:bg-gray-700 transition-colors"
        >
          <svg className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Refresh
        </button>
      </div>

      {/* Overall status banner */}
      {error ? (
        <div className="flex items-center gap-3 bg-red-600/10 border border-red-600/30 rounded-lg px-4 py-4">
          <div className="w-3 h-3 rounded-full bg-red-500 flex-shrink-0 animate-pulse" />
          <div>
            <p className="text-sm font-semibold text-red-400">System unreachable</p>
            <p className="text-xs text-red-400/70 mt-0.5">{error}</p>
          </div>
        </div>
      ) : loading && !status ? (
        <div className="flex items-center gap-3 bg-gray-800 border border-gray-700 rounded-lg px-4 py-4">
          <div className="w-3 h-3 rounded-full bg-gray-500 flex-shrink-0" />
          <p className="text-sm text-gray-400">Checking system status...</p>
        </div>
      ) : allOk ? (
        <div className="flex items-center gap-3 bg-green-600/10 border border-green-600/30 rounded-lg px-4 py-4">
          <div className="w-3 h-3 rounded-full bg-green-500 flex-shrink-0 animate-pulse" />
          <div>
            <p className="text-sm font-semibold text-green-400">All systems operational</p>
            <p className="text-xs text-green-400/70 mt-0.5">{services.length} services healthy</p>
          </div>
        </div>
      ) : anyDown ? (
        <div className="flex items-center gap-3 bg-orange-600/10 border border-orange-600/30 rounded-lg px-4 py-4">
          <div className="w-3 h-3 rounded-full bg-orange-500 flex-shrink-0 animate-pulse" />
          <div>
            <p className="text-sm font-semibold text-orange-400">Partial outage detected</p>
            <p className="text-xs text-orange-400/70 mt-0.5">
              {services.filter((s) => s.status === 'down').length} service(s) down
            </p>
          </div>
        </div>
      ) : null}

      {/* Service cards */}
      <div className="space-y-2">
        {services.map((svc) => (
          <ServiceCard
            key={svc.name}
            name={svc.name}
            status={svc.status}
            detail={svc.detail}
          />
        ))}
      </div>

      {/* Raw JSON (for debugging) */}
      {status && (
        <details className="group">
          <summary className="cursor-pointer text-xs text-gray-600 hover:text-gray-400 transition-colors select-none">
            Raw health response
          </summary>
          <pre className="mt-2 bg-gray-800 border border-gray-700 rounded p-4 text-xs text-gray-400 overflow-x-auto">
            {JSON.stringify(status, null, 2)}
          </pre>
        </details>
      )}
    </div>
  )
}
