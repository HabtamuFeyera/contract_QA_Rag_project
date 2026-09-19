"""
Entry point for LexiRAG (ContractAdvisor-AI).
Launches the FastAPI backend service via Uvicorn.
"""

import uvicorn
import os
import sys

# Ensure backend directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "false").lower() in ("true", "1", "yes")
    print(f"🚀 Starting LexiRAG Backend on http://{host}:{port} (reload={reload})")
    uvicorn.run("src.app:app", host=host, port=port, reload=reload)
