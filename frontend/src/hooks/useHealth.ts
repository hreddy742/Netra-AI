'use client'

import { useState, useEffect, useCallback } from 'react'
import { getHealth } from '@/lib/api'
import type { HealthStatus } from '@/types'

interface HealthState {
  status: HealthStatus | null
  loading: boolean
  error: string | null
  lastChecked: Date | null
}

export function useHealth(intervalMs = 5000) {
  const [state, setState] = useState<HealthState>({
    status: null,
    loading: true,
    error: null,
    lastChecked: null,
  })

  const check = useCallback(async () => {
    try {
      const status = await getHealth()
      setState({
        status,
        loading: false,
        error: null,
        lastChecked: new Date(),
      })
    } catch (err) {
      setState((prev) => ({
        ...prev,
        loading: false,
        error: err instanceof Error ? err.message : 'Health check failed',
        lastChecked: new Date(),
      }))
    }
  }, [])

  useEffect(() => {
    check()
    const interval = setInterval(check, intervalMs)
    return () => clearInterval(interval)
  }, [check, intervalMs])

  return { ...state, refresh: check }
}
