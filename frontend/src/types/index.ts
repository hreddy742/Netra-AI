export interface Incident {
  id: string
  camera_id: string
  track_id: number
  risk_score: number
  fsm_score: number
  shopformer_score: number
  theft_stage: TheftStage
  concealment_type: string
  severity: Severity
  status: IncidentStatus
  occurred_at: string
  operator_notes?: string
}

export type TheftStage =
  | 'BROWSING'
  | 'NEAR_SHELF'
  | 'SHELF_INTERACTION'
  | 'ITEM_PICKED'
  | 'CONCEALMENT'
  | 'HIGH_RISK_EXIT'

export type Severity = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'

export type IncidentStatus = 'OPEN' | 'REVIEWING' | 'CONFIRMED' | 'DISMISSED'

export interface Camera {
  id: string
  display_name: string
  location: string
  rtsp_url: string
  enabled: boolean
}

export interface IncidentListResponse {
  total: number
  incidents: Incident[]
}

export interface IncidentFilters {
  severity?: Severity
  camera_id?: string
  status?: IncidentStatus
  limit?: number
  offset?: number
}

export interface ReviewPayload {
  verdict: 'CONFIRMED' | 'FALSE_POSITIVE' | 'DISMISSED'
  notes?: string
}

export interface HealthStatus {
  service: 'ok' | 'down'
  [key: string]: string
}

export interface RiskScoreEvent {
  camera_id: string
  risk_score: number
  timestamp: string
}

export interface IncidentSSEEvent {
  type: 'new_incident' | 'incident_updated'
  incident: Incident
}

export interface CameraFormData {
  display_name: string
  location: string
  rtsp_url: string
  enabled: boolean
}
