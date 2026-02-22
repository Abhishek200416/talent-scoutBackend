"""
Root-level WSGI entry point for Gunicorn
Required when using: gunicorn app:app
"""

from backend.server import app

# These must be present for gunicorn to find the app
application = app
app = app
