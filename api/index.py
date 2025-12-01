# vercel runtime python3.12
import sys
from pathlib import Path

# Add parent directory to path so we can import main
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app

# Export the app as the handler
handler = app
