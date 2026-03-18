'use client'

interface RiskBarProps {
  score: number // 0-1
  showLabel?: boolean
  height?: string
}

function getRiskColor(score: number): string {
  if (score < 0.33) return 'bg-green-500'
  if (score < 0.66) return 'bg-yellow-500'
  if (score < 0.85) return 'bg-orange-500'
  return 'bg-red-500'
}

function getRiskLabel(score: number): string {
  if (score < 0.33) return 'LOW'
  if (score < 0.66) return 'MEDIUM'
  if (score < 0.85) return 'HIGH'
  return 'CRITICAL'
}

export function RiskBar({ score, showLabel = false, height = 'h-2' }: RiskBarProps) {
  const pct = Math.min(100, Math.max(0, score * 100))
  const color = getRiskColor(score)

  return (
    <div className="flex items-center gap-2 w-full">
      <div className={`flex-1 bg-gray-700 rounded-full ${height} overflow-hidden`}>
        <div
          className={`${color} ${height} rounded-full transition-all duration-300`}
          style={{ width: `${pct}%` }}
        />
      </div>
      {showLabel && (
        <span className="text-xs text-gray-400 w-16 text-right">
          {getRiskLabel(score)} {(score * 100).toFixed(0)}%
        </span>
      )}
    </div>
  )
}
