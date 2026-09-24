# Netra AI

A microservices platform for real-time retail video analytics — person tracking, behavior analysis, and shoplifting/theft-risk detection from live camera feeds.

Netra AI is the third-generation architecture in this lineage: it started as a single-script prototype ([`edgeguard-mvp`](https://github.com/hreddy742/edgeguard-mvp)), matured into a monolithic FastAPI+Streamlit app (`edgeguard/` inside [`EDGE_AI`](https://github.com/hreddy742/EDGE_AI)), and was rebuilt here as a distributed, service-oriented system with a dedicated behavior-classification model (`shopformer`).

## Architecture

10 backend services communicate over Redis, with shared state in PostgreSQL, fronted by a FastAPI gateway and a Next.js dashboard:

```
Camera / video source
        |
   mediabridge   -- ingest, shared-memory frame distribution (IPC host mode)
        |
   inference     -- GPU-backed detection/pose model (nvidia runtime)
        |
   association   -- links detections into per-person tracks across frames
        |
   behavior      -- behavioral feature extraction from tracks
        |
   shopformer    -- transformer-based theft/risk sequence classifier
        |
   risk          -- risk scoring and thresholding
        |
   alerting      -- alert dispatch
        |
   persistence   -- writes tracks/events/alerts to PostgreSQL
        |
   gateway (FastAPI + WebSocket) -- REST + realtime API
        |
   frontend (Next.js) -- live dashboard
```

Redis carries inter-service messaging; Prometheus + Grafana (bundled in `docker-compose.yml`) provide metrics and dashboards; `redis-exporter` feeds Redis metrics to Prometheus.

## `shopformer`

The behavior-classification model at the core of the risk pipeline: `backend/services/shopformer/` contains its own `dataset.py`, `model.py`, `train.py`, and `export.py` — i.e. this isn't a wrapper around a third-party model, the classifier is trained and exported from within this repo.

## Tech stack

| Layer | Technology |
| --- | --- |
| Backend services | Python, Redis pub/sub, PostgreSQL |
| Inference | GPU-accelerated (NVIDIA runtime), custom `shopformer` transformer model |
| API | FastAPI + WebSocket gateway |
| Frontend | Next.js, TypeScript, Tailwind CSS |
| Observability | Prometheus, Grafana, redis-exporter |
| Orchestration | Docker Compose (`docker-compose.yml` full stack, `docker-compose.edge.yml` edge deployment) |

## Getting started

```bash
cp .env.example .env
docker compose up -d
```

This starts Redis, PostgreSQL, all 9 backend services, the FastAPI gateway (`:8000`), the Next.js frontend (`:3000`), and the observability stack (Prometheus `:9091`, Grafana `:3001`).

The `inference` service requests an NVIDIA GPU via Docker's device reservations — a CUDA-capable host and the NVIDIA Container Toolkit are required for that service to start.

## Status

This is an early-stage rebuild — the repository does not yet have commit history showing iterative development, so treat this as an initial architecture rather than a battle-tested system. If you're evaluating it, look at `backend/services/shopformer/` for the model implementation and `backend/gateway/` for the API surface first.
