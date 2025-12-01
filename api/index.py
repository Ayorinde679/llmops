# vercel runtime python3.12
import sys
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import and return the FastAPI app
from main import app
