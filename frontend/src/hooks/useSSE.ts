'use client'

import { useEffect, useRef, useCallback } from 'react'

interface SSEOptions<T> {
  url: string
  onMessage: (data: T) => void
  onError?: (err: Event) => void
  enabled?: boolean
}

export function useSSE<T>({
  url,
  onMessage,
  onError,
  enabled = true,
}: SSEOptions<T>) {
  const esRef = useRef<EventSource | null>(null)
  const onMessageRef = useRef(onMessage)
  const onErrorRef = useRef(onError)

  // Keep refs current without re-triggering effect
  onMessageRef.current = onMessage
  onErrorRef.current = onError

  const connect = useCallback(() => {
    if (esRef.current) {
      esRef.current.close()
    }

    const es = new EventSource(url)
    esRef.current = es

    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as T
        onMessageRef.current(data)
      } catch {
        // Non-JSON SSE message, ignore
      }
    }

    es.onerror = (err) => {
      onErrorRef.current?.(err)
      // Auto-reconnect after 3s on error
      es.close()
      esRef.current = null
      setTimeout(() => {
        if (enabled) connect()
      }, 3000)
    }

    return es
  }, [url, enabled])

  useEffect(() => {
    if (!enabled) return

    connect()

    return () => {
      esRef.current?.close()
      esRef.current = null
    }
  }, [connect, enabled])
}
