# Simple test to check what's causing the Vercel crash
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    print("Testing imports...")
    import flask
    print("✓ Flask imported")
    
    import flask_sqlalchemy
    print("✓ Flask-SQLAlchemy imported")
    
    import flask_login
    print("✓ Flask-Login imported")
    
    import google.generativeai
    print("✓ Google Generative AI imported")
    
    print("Testing app import...")
    from app import app
    print("✓ App imported successfully")
    
    print("All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
