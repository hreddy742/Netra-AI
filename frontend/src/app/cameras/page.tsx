'use client'

import { useState } from 'react'
import { useCameras } from '@/hooks/useCameras'
import type { CameraFormData } from '@/types'

const EMPTY_FORM: CameraFormData = {
  display_name: '',
  location: '',
  rtsp_url: '',
  enabled: true,
}

function CameraStatusDot({ enabled }: { enabled: boolean }) {
  return (
    <div className="flex items-center gap-1.5">
      <div
        className={`w-2 h-2 rounded-full ${
          enabled ? 'bg-green-500 animate-pulse' : 'bg-gray-600'
        }`}
      />
      <span className={`text-xs ${enabled ? 'text-green-400' : 'text-gray-500'}`}>
        {enabled ? 'Active' : 'Disabled'}
      </span>
    </div>
  )
}

export default function CamerasPage() {
  const { cameras, loading, error, addCamera, refetch } = useCameras()
  const [showForm, setShowForm] = useState(false)
  const [form, setForm] = useState<CameraFormData>(EMPTY_FORM)
  const [formError, setFormError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  function updateForm(field: keyof CameraFormData, value: string | boolean) {
    setForm((prev) => ({ ...prev, [field]: value }))
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!form.display_name.trim() || !form.rtsp_url.trim()) {
      setFormError('Display name and RTSP URL are required.')
      return
    }
    setSubmitting(true)
    setFormError(null)
    try {
      await addCamera(form)
      setForm(EMPTY_FORM)
      setShowForm(false)
    } catch (err) {
      setFormError(err instanceof Error ? err.message : 'Failed to add camera')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="p-6 space-y-6 max-w-4xl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-gray-100">Cameras</h1>
          <p className="text-sm text-gray-500 mt-0.5">
            {cameras.length} camera{cameras.length !== 1 ? 's' : ''} configured
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={refetch}
            className="flex items-center gap-2 px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm text-gray-300 hover:bg-gray-700 transition-colors"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Refresh
          </button>
          <button
            onClick={() => setShowForm((v) => !v)}
            className="flex items-center gap-2 px-3 py-2 bg-red-600 hover:bg-red-700 rounded text-sm text-white font-medium transition-colors"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            Add Camera
          </button>
        </div>
      </div>

      {/* Add camera form */}
      {showForm && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-5">
          <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wide mb-4">
            New Camera
          </h2>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <label className="text-xs text-gray-500 font-medium">Display Name *</label>
                <input
                  type="text"
                  value={form.display_name}
                  onChange={(e) => updateForm('display_name', e.target.value)}
                  placeholder="e.g. Entrance Camera 1"
                  className="w-full bg-gray-900 border border-gray-700 rounded text-sm text-gray-200 placeholder-gray-600 px-3 py-2 focus:outline-none focus:border-gray-500 transition-colors"
                />
              </div>
              <div className="space-y-1.5">
                <label className="text-xs text-gray-500 font-medium">Location</label>
                <input
                  type="text"
                  value={form.location}
                  onChange={(e) => updateForm('location', e.target.value)}
                  placeholder="e.g. Aisle 3, Entrance"
                  className="w-full bg-gray-900 border border-gray-700 rounded text-sm text-gray-200 placeholder-gray-600 px-3 py-2 focus:outline-none focus:border-gray-500 transition-colors"
                />
              </div>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs text-gray-500 font-medium">RTSP URL *</label>
              <input
                type="text"
                value={form.rtsp_url}
                onChange={(e) => updateForm('rtsp_url', e.target.value)}
                placeholder="rtsp://user:pass@192.168.1.x:554/stream"
                className="w-full bg-gray-900 border border-gray-700 rounded text-sm text-gray-200 placeholder-gray-600 px-3 py-2 focus:outline-none focus:border-gray-500 font-mono transition-colors"
              />
            </div>

            <div className="flex items-center gap-3">
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={form.enabled}
                  onChange={(e) => updateForm('enabled', e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-9 h-5 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-green-600" />
              </label>
              <span className="text-sm text-gray-400">Enable camera immediately</span>
            </div>

            {formError && (
              <p className="text-xs text-red-400 bg-red-400/10 border border-red-400/20 rounded px-3 py-2">
                {formError}
              </p>
            )}

            <div className="flex gap-2 pt-1">
              <button
                type="submit"
                disabled={submitting}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-700 disabled:text-gray-500 text-white text-sm font-medium rounded transition-colors flex items-center gap-2"
              >
                {submitting ? (
                  <>
                    <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Adding...
                  </>
                ) : (
                  'Add Camera'
                )}
              </button>
              <button
                type="button"
                onClick={() => { setShowForm(false); setForm(EMPTY_FORM); setFormError(null) }}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-300 text-sm rounded transition-colors"
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="flex items-center gap-3 bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3 text-red-400 text-sm">
          <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.072 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
          {error}
        </div>
      )}

      {/* Camera list */}
      {loading ? (
        <div className="space-y-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <div key={i} className="bg-gray-800 border border-gray-700 rounded-lg p-4 h-20 animate-pulse" />
          ))}
        </div>
      ) : cameras.length === 0 ? (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-12 text-center">
          <svg className="w-10 h-10 text-gray-600 mx-auto mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
          </svg>
          <p className="text-gray-400 font-medium">No cameras configured</p>
          <p className="text-gray-600 text-sm mt-1">Click "Add Camera" to get started</p>
        </div>
      ) : (
        <div className="space-y-2">
          {cameras.map((camera) => (
            <div
              key={camera.id}
              className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-3.5 flex items-center gap-4"
            >
              <div className="w-10 h-10 rounded-lg bg-gray-700 flex items-center justify-center flex-shrink-0">
                <svg className="w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
                </svg>
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <p className="text-sm font-semibold text-gray-100 truncate">{camera.display_name}</p>
                  <CameraStatusDot enabled={camera.enabled} />
                </div>
                <p className="text-xs text-gray-500 truncate">{camera.location}</p>
                <p className="text-xs text-gray-600 font-mono truncate mt-0.5">{camera.rtsp_url}</p>
              </div>

              <div className="text-xs text-gray-600 font-mono flex-shrink-0">
                {camera.id.slice(0, 8)}...
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
