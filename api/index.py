# Vercel serverless function entry point
import sys
import os
import traceback

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set Vercel environment variable
os.environ['VERCEL'] = '1'

try:
    # Import the Flask app from app.py
    from app import app
    
    # Export the app for Vercel
    application = app
    
    # Add comprehensive error handling
    @application.errorhandler(Exception)
    def handle_exception(e):
        """Handle all exceptions and return proper error response"""
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        
        # Log the error (Vercel will capture this)
        print(f"ERROR: {error_msg}")
        print(f"TRACEBACK: {error_traceback}")
        
        # Return a proper error response
        return {
            'error': 'Internal Server Error',
            'message': 'An error occurred while processing your request',
            'status': 500,
            'details': error_msg if os.environ.get('FLASK_ENV') == 'development' else 'Contact support for assistance'
        }, 500

    # Add a simple test route
    @application.route('/test')
    def test_route():
        return {'status': 'ok', 'message': 'API is working'}

    if __name__ == '__main__':
        app.run()
        
except Exception as e:
    # If there's an error importing the app, create a minimal error handler
    print(f"CRITICAL ERROR: Failed to import app: {str(e)}")
    print(f"TRACEBACK: {traceback.format_exc()}")
    
    from flask import Flask, jsonify
    error_app = Flask(__name__)
    
    @error_app.route('/')
    def error_handler():
        return jsonify({
            'error': 'Application Error',
            'message': 'Failed to initialize application',
            'status': 500
        }), 500
    
    @error_app.route('/test')
    def test_error():
        return jsonify({'status': 'error', 'message': 'App failed to load'})
    
    application = error_app