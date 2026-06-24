import os
import sys
from pathlib import Path

# Ensure the Django project directory is on the import path for Vercel.
project_dir = Path(__file__).resolve().parent.parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Speech.settings")
os.environ.setdefault("VERCEL", "1")

from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()
app = application
