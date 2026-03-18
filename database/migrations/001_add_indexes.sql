-- Migration 001: Add missing performance indexes
-- Apply to existing deployments: psql $DATABASE_URL -f 001_add_indexes.sql
-- Idempotent: uses IF NOT EXISTS throughout

CREATE INDEX IF NOT EXISTS idx_incidents_camera_created
    ON incidents(camera_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_incidents_track
    ON incidents(track_id);

CREATE INDEX IF NOT EXISTS idx_evidence_frames_incident
    ON evidence_frames(incident_id);

CREATE INDEX IF NOT EXISTS idx_reviews_incident
    ON reviews(incident_id);
