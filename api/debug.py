# Debug script for Vercel deployment issues
import sys
import os
import traceback

print("=== VERCEL DEBUG SCRIPT ===")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Check environment variables
print("\n=== ENVIRONMENT VARIABLES ===")
env_vars = ['SECRET_KEY', 'GOOGLE_AI_API_KEY', 'DATABASE_URL']
for var in env_vars:
    value = os.environ.get(var)
    if value:
        # Mask sensitive data
        if 'KEY' in var or 'SECRET' in var:
            masked_value = value[:8] + '...' + value[-4:] if len(value) > 12 else '***'
        else:
            masked_value = value[:50] + '...' if len(value) > 50 else value
        print(f"{var}: {masked_value}")
    else:
        print(f"{var}: NOT SET")

# Check file structure
print("\n=== FILE STRUCTURE ===")
files_to_check = ['app.py', 'requirements.txt', 'vercel.json']
for file in files_to_check:
    if os.path.exists(file):
        print(f"✓ {file} exists")
    else:
        print(f"✗ {file} missing")

# Check imports
print("\n=== IMPORT TESTS ===")
try:
    import flask
    print("✓ Flask imported successfully")
except Exception as e:
    print(f"✗ Flask import failed: {e}")

try:
    import flask_sqlalchemy
    print("✓ Flask-SQLAlchemy imported successfully")
except Exception as e:
    print(f"✗ Flask-SQLAlchemy import failed: {e}")

try:
    import google.generativeai
    print("✓ Google Generative AI imported successfully")
except Exception as e:
    print(f"✗ Google Generative AI import failed: {e}")

try:
    import psycopg2
    print("✓ psycopg2 imported successfully")
except Exception as e:
    print(f"✗ psycopg2 import failed: {e}")

# Try to import the main app
print("\n=== APP IMPORT TEST ===")
try:
    from app import app
    print("✓ App imported successfully")
    
    # Test database connection
    try:
        with app.app_context():
            db_url = app.config.get('SQLALCHEMY_DATABASE_URI', 'Not set')
            print(f"Database URL: {db_url[:50]}..." if len(db_url) > 50 else f"Database URL: {db_url}")
    except Exception as e:
        print(f"✗ Database configuration error: {e}")
        
except Exception as e:
    print(f"✗ App import failed: {e}")
    print(f"Traceback: {traceback.format_exc()}")

print("\n=== DEBUG COMPLETE ===")
