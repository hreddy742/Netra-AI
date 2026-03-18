import type {
  Camera,
  CameraFormData,
  HealthStatus,
  Incident,
  IncidentFilters,
  IncidentListResponse,
  ReviewPayload,
} from '@/types'

const BASE = ''

async function request<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
    ...options,
  })

  if (!res.ok) {
    const text = await res.text().catch(() => 'Unknown error')
    throw new Error(`API error ${res.status}: ${text}`)
  }

  const contentType = res.headers.get('content-type') || ''
  if (contentType.includes('application/json')) {
    return res.json() as Promise<T>
  }
  return res.text() as unknown as Promise<T>
}

// Auth
export async function getAuthToken(username: string, password: string) {
  return request<{ access_token: string; token_type: string }>(
    `/api/auth/login`,
    {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    }
  )
}

// Cameras
export async function getCameras(): Promise<Camera[]> {
  return request<Camera[]>('/api/cameras')
}

export async function createCamera(data: CameraFormData): Promise<Camera> {
  return request<Camera>('/api/cameras', {
    method: 'POST',
    body: JSON.stringify(data),
  })
}

// Incidents
export async function getIncidents(
  filters: IncidentFilters = {}
): Promise<IncidentListResponse> {
  const params = new URLSearchParams()
  if (filters.severity) params.set('severity', filters.severity)
  if (filters.camera_id) params.set('camera_id', filters.camera_id)
  if (filters.status) params.set('status', filters.status)
  if (filters.limit !== undefined) params.set('limit', String(filters.limit))
  if (filters.offset !== undefined) params.set('offset', String(filters.offset))

  const qs = params.toString()
  return request<IncidentListResponse>(`/api/incidents${qs ? `?${qs}` : ''}`)
}

export async function getIncident(id: string): Promise<Incident> {
  return request<Incident>(`/api/incidents/${id}`)
}

export async function reviewIncident(
  id: string,
  payload: ReviewPayload
): Promise<Incident> {
  return request<Incident>(`/api/incidents/${id}/review`, {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

// Health
export async function getHealth(): Promise<HealthStatus> {
  return request<HealthStatus>('/api/health')
}

// Stream URLs (used as img src / EventSource)
export function getMjpegStreamUrl(cameraId: string): string {
  return `/api/stream/${cameraId}`
}

export function getRiskStreamUrl(cameraId: string): string {
  return `/api/stream/risk/${cameraId}`
}

export function getIncidentStreamUrl(): string {
  return `/api/stream/incidents`
}
