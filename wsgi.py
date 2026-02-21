"""
WSGI entry point for Gunicorn
Render deployment uses this to start the application
"""
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server import app

# Expose the WSGI app callable for Gunicorn
application = app
app_instance = app
app = app  # For gunicorn compatibility
