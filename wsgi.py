"""
WSGI entry point for Gunicorn
Render deployment uses this to start the application
"""

import sys
from pathlib import Path

# Add the backend directory to Python path for imports
backend_path = Path(__file__).parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from server import app

# Expose the WSGI app callable for Gunicorn
application = app
app = app
