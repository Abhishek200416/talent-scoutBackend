"""
Backend app entry point for Gunicorn
Required when using: gunicorn backend.app:app
"""

import sys
from pathlib import Path

# Add the backend directory to Python path for imports
backend_path = Path(__file__).parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from server import app

# These must be present for gunicorn to find the app
application = app
app = app
