# vercel runtime python3.12
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app

# For Vercel serverless, we need to handle the ASGI app properly
# Vercel will call this as the handler
def handler(request):
    """This is called by Vercel for each request"""
    # For ASGI apps on Vercel, we need the Mangum adapter
    from mangum import Mangum
    
    asgi_handler = Mangum(app)
    return asgi_handler(request)
