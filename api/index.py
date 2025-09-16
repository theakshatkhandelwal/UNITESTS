# Vercel serverless function entry point
# Import the main Flask app from the parent directory
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Flask app from app.py
from app import app

# Export the app for Vercel
application = app

if __name__ == '__main__':
    app.run()