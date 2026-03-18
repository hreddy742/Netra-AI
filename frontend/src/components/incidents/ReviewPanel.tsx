'use client'

import { useState } from 'react'
import type { Incident } from '@/types'
import { reviewIncident } from '@/lib/api'

interface ReviewPanelProps {
  incident: Incident
  onReviewed: (updated: Incident) => void
}

type Verdict = 'CONFIRMED' | 'FALSE_POSITIVE' | 'DISMISSED'

const VERDICT_CONFIG: Record<Verdict, { label: string; className: string; activeClassName: string }> = {
  CONFIRMED: {
    label: 'Confirm Theft',
    className: 'border-red-600/40 text-red-400 hover:bg-red-600/10',
    activeClassName: 'bg-red-600 border-red-600 text-white',
  },
  FALSE_POSITIVE: {
    label: 'False Positive',
    className: 'border-green-600/40 text-green-400 hover:bg-green-600/10',
    activeClassName: 'bg-green-600 border-green-600 text-white',
  },
  DISMISSED: {
    label: 'Dismiss',
    className: 'border-gray-600/40 text-gray-400 hover:bg-gray-700',
    activeClassName: 'bg-gray-600 border-gray-600 text-white',
  },
}

export function ReviewPanel({ incident, onReviewed }: ReviewPanelProps) {
  const [verdict, setVerdict] = useState<Verdict | null>(null)
  const [notes, setNotes] = useState(incident.operator_notes ?? '')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)

  const isReviewed = incident.status === 'CONFIRMED' || incident.status === 'DISMISSED'

  async function handleSubmit() {
    if (!verdict) return
    setSubmitting(true)
    setError(null)
    try {
      const updated = await reviewIncident(incident.id, { verdict, notes: notes.trim() || undefined })
      setSuccess(true)
      onReviewed(updated)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Review failed')
    } finally {
      setSubmitting(false)
    }
  }

  if (success) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-5">
        <div className="flex items-center gap-3 text-green-400">
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span className="font-medium">Review submitted successfully</span>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-5 space-y-4">
      <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">
        Operator Review
      </h3>

      {isReviewed && (
        <div className="text-xs text-yellow-400 bg-yellow-400/10 border border-yellow-400/20 rounded px-3 py-2">
          This incident has already been reviewed (status: {incident.status}).
          You can still update it below.
        </div>
      )}

      {/* Verdict buttons */}
      <div className="grid grid-cols-3 gap-2">
        {(Object.keys(VERDICT_CONFIG) as Verdict[]).map((v) => {
          const cfg = VERDICT_CONFIG[v]
          const isActive = verdict === v
          return (
            <button
              key={v}
              onClick={() => setVerdict(isActive ? null : v)}
              className={`px-3 py-2.5 rounded border text-sm font-medium transition-colors ${
                isActive ? cfg.activeClassName : cfg.className
              }`}
            >
              {cfg.label}
            </button>
          )
        })}
      </div>

      {/* Notes */}
      <div className="space-y-1.5">
        <label className="text-xs text-gray-500 font-medium">Operator Notes (optional)</label>
        <textarea
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          rows={3}
          placeholder="Add investigation notes, observations, or follow-up actions..."
          className="w-full bg-gray-900 border border-gray-700 rounded text-sm text-gray-200 placeholder-gray-600 px-3 py-2 focus:outline-none focus:border-gray-500 resize-none"
        />
      </div>

      {error && (
        <p className="text-xs text-red-400 bg-red-400/10 border border-red-400/20 rounded px-3 py-2">
          {error}
        </p>
      )}

      <button
        onClick={handleSubmit}
        disabled={!verdict || submitting}
        className="w-full py-2.5 rounded bg-red-600 hover:bg-red-700 disabled:bg-gray-700 disabled:text-gray-500 text-white font-medium text-sm transition-colors flex items-center justify-center gap-2"
      >
        {submitting ? (
          <>
            <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Submitting...
          </>
        ) : (
          'Submit Review'
        )}
      </button>
    </div>
  )
}
