#!/usr/bin/env python
"""
Script pour lancer l'API FastAPI.

Usage:
    python scripts/run_api.py

Ou avec uvicorn directement:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import uvicorn
import sys
from pathlib import Path

# Ajout du chemin au PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Mode développement (désactiver en production)
    )
