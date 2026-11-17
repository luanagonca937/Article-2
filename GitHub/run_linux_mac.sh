#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-tested.txt || pip install -r requirements.txt
python reproduce_article2.py
