#!/usr/bin/env bash
set -e
uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload
