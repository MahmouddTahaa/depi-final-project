#!/usr/bin/zsh 

# ChatBoot launcher — Linux / macOS
set -e

cd "$(dirname "$0")"

# Create venv if missing
if [ ! -d "venv" ]; then
  echo "Creating virtual env…"
  python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install deps if not yet installed
if [ ! -f "venv/.installed" ]; then
  echo "Installing dependencies (first run, ~5 min)…"
  pip install --upgrade pip
  pip install -r requirements.txt
  python -m spacy download en_core_web_sm
  touch venv/.installed
fi

echo ""
echo "▶ Launching ChatBoot at http://localhost:8501"
echo ""
streamlit run streamlit_app.py
