# Standalone Multilingual Language Classifier Demo

This folder is separate from AURA. It contains its own FastAPI application, dependency list, environment configuration, systemd unit example, static frontend and API routes.

## Layout

- backend/app.py — CPU inference API
- backend/requirements.txt — isolated Python dependencies
- backend/.env.example — model path and CORS settings
- backend/classifier-api.service.example — separate systemd service on 127.0.0.1:8011
- frontend/ — static black-and-white website for Cloudflare Pages

## Backend setup

Create a separate project directory and virtual environment on EC2. Copy this backend into /opt/multilingual-classifier/backend, then:

    python3 -m venv /opt/multilingual-classifier/venv
    /opt/multilingual-classifier/venv/bin/pip install -r requirements.txt

Copy .env.example to .env and set MODEL_DIR to the extracted Hugging Face checkpoint directory. The directory must contain config.json, tokenizer files and model weights. The API does not download weights and no model artifacts are committed here.

Start for a local test:

    set -a && source .env && set +a
    /opt/multilingual-classifier/venv/bin/uvicorn app:app --host 127.0.0.1 --port 8011 --workers 1

Endpoints:
- GET /api/health
- GET /api/languages
- POST /api/classify with JSON body {"text":"Mujhe kal meeting ke liye Delhi jaana hai."}

The service stays available for health diagnostics if model loading fails; classification returns HTTP 503 rather than fabricated output.

## Cloudflare Pages frontend

Deploy demo/frontend as the Pages project root. It is a static site and needs no build command. Set CLASSIFIER_API_BASE in frontend/config.js to the public HTTPS origin that will route to this API. Add the exact Pages origin to backend CORS_ORIGINS.

## EC2 isolation

Use a separate directory, venv, systemd unit and local port 8011. The example service binds only to 127.0.0.1. It does not modify AURA's aura.service, port 8000, Nginx config, database, routes or Cloudflare Worker.

After deploying the API, verify it locally with curl http://127.0.0.1:8011/api/health. Then configure a separate reverse-proxy route or Cloudflare Tunnel hostname for this service. Inspect the existing tunnel/proxy setup before making infrastructure changes; do not reuse or alter AURA routes. This commit does not change EC2, DNS, Cloudflare Pages, Nginx or tunnels.

## Accuracy and reporting

The repository README describes a 12-label policy including none. The API returns labels from the loaded checkpoint's id2label config. Softmax scores are not calibrated probabilities. Inference time measures tokenization plus the CPU model forward pass; it is not network or full voice-agent latency. The classifier was designed to provide a language tag after STT; this demo does not claim production integration or guaranteed same-turn switching.
