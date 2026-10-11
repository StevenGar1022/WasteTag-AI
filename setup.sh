#!/usr/bin/env bash
# WASTETAG-AI — instalador (envoltura de install.py, multiplataforma).
# Uso: ./setup.sh
set -e
cd "$(dirname "$0")"
if [ -x env/bin/python ]; then
    exec env/bin/python install.py
else
    exec python3 install.py
fi
