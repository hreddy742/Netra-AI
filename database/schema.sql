-- Netra AI — PostgreSQL schema
-- Run once at startup; idempotent via IF NOT EXISTS

-- ----------------------------------------------------------------
-- Cameras
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cameras (
    id            TEXT PRIMARY KEY,
    display_name  TEXT NOT NULL,
    location      TEXT,
    rtsp_url      TEXT,
    source_type   TEXT NOT NULL DEFAULT 'rtsp',
    enabled       BOOLEAN NOT NULL DEFAULT TRUE,
    zones_json    JSONB,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ----------------------------------------------------------------
-- Tracks (persons detected)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tracks (
    id            BIGSERIAL PRIMARY KEY,
    camera_id     TEXT NOT NULL REFERENCES cameras(id),
    track_id      INT NOT NULL,
    first_seen_at TIMESTAMPTZ NOT NULL,
    last_seen_at  TIMESTAMPTZ NOT NULL,
    thumbnail_path TEXT,
    reid_embedding REAL[]
);

CREATE INDEX IF NOT EXISTS idx_tracks_camera ON tracks (camera_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_tracks_cam_track ON tracks (camera_id, track_id);

-- ----------------------------------------------------------------
-- Incidents
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS incidents (
    id                TEXT PRIMARY KEY,
    camera_id         TEXT NOT NULL REFERENCES cameras(id),
    track_id          INT NOT NULL,
    risk_score        REAL NOT NULL,
    fsm_score         REAL NOT NULL,
    shopformer_score  REAL NOT NULL,
    theft_stage       TEXT NOT NULL,
    concealment_type  TEXT NOT NULL,
    severity          TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'OPEN',
    model_version     TEXT,
    logic_version     TEXT,
    operator_notes    TEXT,
    reviewed_by       TEXT,
    reviewed_at       TIMESTAMPTZ,
    occurred_at       TIMESTAMPTZ NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_incidents_camera     ON incidents (camera_id);
CREATE INDEX IF NOT EXISTS idx_incidents_occurred   ON incidents (occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_incidents_severity   ON incidents (severity);
CREATE INDEX IF NOT EXISTS idx_incidents_status     ON incidents (status);

-- ----------------------------------------------------------------
-- Multi-store foundation (additive — existing rows default to 'default')
-- ----------------------------------------------------------------
ALTER TABLE cameras   ADD COLUMN IF NOT EXISTS store_id TEXT NOT NULL DEFAULT 'default';
ALTER TABLE cameras   ADD COLUMN IF NOT EXISTS org_id   TEXT NOT NULL DEFAULT 'default';
ALTER TABLE incidents ADD COLUMN IF NOT EXISTS store_id TEXT NOT NULL DEFAULT 'default';
ALTER TABLE incidents ADD COLUMN IF NOT EXISTS org_id   TEXT NOT NULL DEFAULT 'default';

CREATE INDEX IF NOT EXISTS idx_incidents_store ON incidents (store_id, org_id);

-- ----------------------------------------------------------------
-- Behavior signals (per incident)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS behavior_signals (
    id            BIGSERIAL PRIMARY KEY,
    incident_id   TEXT NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    camera_id     TEXT NOT NULL,
    track_id      INT NOT NULL,
    signal_name   TEXT NOT NULL,
    weight        REAL NOT NULL,
    fsm_state     TEXT NOT NULL,
    occurred_at   TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_signals_incident ON behavior_signals (incident_id);

-- ----------------------------------------------------------------
-- Evidence frames
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS evidence_frames (
    id            BIGSERIAL PRIMARY KEY,
    incident_id   TEXT NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    frame_path    TEXT NOT NULL,
    frame_seq     INT,
    occurred_at   TIMESTAMPTZ NOT NULL
);

-- ----------------------------------------------------------------
-- Evidence clips
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS evidence_clips (
    id            BIGSERIAL PRIMARY KEY,
    incident_id   TEXT NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    clip_path     TEXT NOT NULL,
    duration_sec  REAL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ----------------------------------------------------------------
-- Reviews (operator workflow)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reviews (
    id            BIGSERIAL PRIMARY KEY,
    incident_id   TEXT NOT NULL REFERENCES incidents(id),
    operator_id   TEXT NOT NULL,
    verdict       TEXT NOT NULL,    -- CONFIRMED | DISMISSED | ESCALATED
    notes         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ----------------------------------------------------------------
-- Model versions (for auditability)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_versions (
    id              BIGSERIAL PRIMARY KEY,
    model_name      TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    logic_version   TEXT NOT NULL,
    deployed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notes           TEXT
);

-- ----------------------------------------------------------------
-- Users (operators)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'operator',   -- operator | admin
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ----------------------------------------------------------------
-- Phase 9: Training samples (evidence clips tagged for retraining)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS training_samples (
    id            BIGSERIAL PRIMARY KEY,
    incident_id   TEXT NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    camera_id     TEXT NOT NULL,
    track_id      INT NOT NULL,
    clip_path     TEXT,
    label         INT NOT NULL DEFAULT 1,  -- 1=theft, 0=normal
    split         TEXT NOT NULL DEFAULT 'train',  -- train|val|test
    exported_at   TIMESTAMPTZ,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_training_label ON training_samples (label, split);

-- ----------------------------------------------------------------
-- Phase 9: Heatmap snapshots (daily aggregates per camera)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS heatmap_snapshots (
    id          BIGSERIAL PRIMARY KEY,
    camera_id   TEXT NOT NULL,
    store_id    TEXT NOT NULL DEFAULT 'default',
    date        DATE NOT NULL,
    grid_json   JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (camera_id, date)
);

-- ----------------------------------------------------------------
-- Phase 10: Stores (multi-tenant onboarding)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stores (
    store_id     TEXT PRIMARY KEY,
    org_id       TEXT NOT NULL,
    display_name TEXT NOT NULL,
    address      TEXT,
    timezone     TEXT NOT NULL DEFAULT 'UTC',
    plan         TEXT NOT NULL DEFAULT 'standard',  -- standard | enterprise
    active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stores_org ON stores (org_id);

-- ----------------------------------------------------------------
-- Phase 10: Alert escalations
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS alert_escalations (
    id          BIGSERIAL PRIMARY KEY,
    incident_id TEXT NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    camera_id   TEXT NOT NULL,
    reason      TEXT NOT NULL DEFAULT 'Unreviewed after threshold',
    escalated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (incident_id)
);

CREATE INDEX IF NOT EXISTS idx_escalations_cam ON alert_escalations (camera_id);

-- ----------------------------------------------------------------
-- Performance indexes (added phase-1 fix)
-- ----------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_incidents_camera_created
    ON incidents(camera_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_incidents_track
    ON incidents(track_id);
CREATE INDEX IF NOT EXISTS idx_evidence_frames_incident
    ON evidence_frames(incident_id);
CREATE INDEX IF NOT EXISTS idx_reviews_incident
    ON reviews(incident_id);

-- ----------------------------------------------------------------
-- Phase 10: Model deployments (versioned, replaces model_versions)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_deployments (
    id             BIGSERIAL PRIMARY KEY,
    model_name     TEXT NOT NULL,
    model_version  TEXT NOT NULL,
    logic_version  TEXT NOT NULL,
    path           TEXT NOT NULL,
    is_active      BOOLEAN NOT NULL DEFAULT FALSE,
    notes          TEXT,
    deployed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_model_dep_name ON model_deployments (model_name, is_active);

-- ----------------------------------------------------------------
-- Phase 10: Camera calibrations (persisted threshold overrides)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS camera_calibrations (
    id           BIGSERIAL PRIMARY KEY,
    camera_id    TEXT NOT NULL UNIQUE,
    threshold    REAL NOT NULL,
    f_score      REAL,
    sample_count INT,
    calibrated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
