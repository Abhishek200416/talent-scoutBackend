"""
WSGI entry point for Gunicorn
Render deployment uses this to start the application
"""

from .server import app

# Expose the WSGI app callable for Gunicorn
application = app
