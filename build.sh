#!/usr/bin/env bash
# Render native Python build (alternative to Docker — use if switching runtime to python)
set -e
pip install --upgrade pip
pip install -r requirements.txt
