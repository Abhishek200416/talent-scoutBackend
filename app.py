"""
Backend app entry point for Gunicorn
Required when using: gunicorn backend.app:app
"""

import sys
from pathlib import Path

# Add the parent directory (project root) to Python path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.server import app

# These must be present for gunicorn to find the app
application = app
app = app
