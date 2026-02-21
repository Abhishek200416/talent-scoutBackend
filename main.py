# Main entry point for Railway deployment
# Imports and re-exports from server.py for ASGI compatibility
from server import app

# Re-export for uvicorn
__all__ = ['app']
