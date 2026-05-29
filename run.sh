#!/usr/bin/env bash
set -euo pipefail

: '@ 2026 Developed by Saksham Nirula'

if [[ -d ".venv" ]]; then
  source .venv/bin/activate
elif [[ -d "venv" ]]; then
  source venv/bin/activate
else
  echo "No virtual environment found. Create one with:"
  echo "  python3 -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  pip install -r requirements.txt && pip install -e ."
  exit 1
fi

streamlit run src/insightlens/ui/streamlit_app.py
