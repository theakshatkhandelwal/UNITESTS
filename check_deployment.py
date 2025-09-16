#!/usr/bin/env python3
"""
UniTest Deployment Status Checker
This script checks if your deployed application is working correctly.
"""

import requests
import sys
import time
from urllib.parse import urlparse

def check_url(url):
    """Check if a URL is accessible and responding."""
    try:
        print(f"🔍 Checking {url}...")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ {url} is accessible (Status: {response.status_code})")
            return True
        else:
            print(f"⚠️  {url} returned status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ {url} is not accessible: {e}")
        return False

def check_health_endpoints(base_url):
    """Check various endpoints of the application."""
    endpoints = [
        "/",  # Home page
        "/signup",  # Signup page
        "/login",  # Login page
    ]
    
    print(f"\n🏥 Checking application health at {base_url}")
    print("=" * 50)
    
    all_healthy = True
    for endpoint in endpoints:
        full_url = base_url.rstrip('/') + endpoint
        if not check_url(full_url):
            all_healthy = False
    
    return all_healthy

def main():
    """Main function to check deployment status."""
    print("🎓 UniTest Deployment Status Checker")
    print("=" * 40)
    
    if len(sys.argv) != 2:
        print("Usage: python check_deployment.py <your-app-url>")
        print("Example: python check_deployment.py https://your-unittest-app.herokuapp.com")
        return
    
    app_url = sys.argv[1]
    
    # Validate URL format
    try:
        parsed = urlparse(app_url)
        if not parsed.scheme or not parsed.netloc:
            print("❌ Invalid URL format. Please provide a complete URL including http:// or https://")
            return
    except Exception as e:
        print(f"❌ URL validation failed: {e}")
        return
    
    print(f"🎯 Checking deployment status for: {app_url}")
    
    # Check if the main application is accessible
    if check_health_endpoints(app_url):
        print("\n🎉 Congratulations! Your UniTest application is deployed and working!")
        print("\n📋 Deployment Summary:")
        print(f"   ✅ Application URL: {app_url}")
        print(f"   ✅ Home page: Accessible")
        print(f"   ✅ Signup page: Accessible")
        print(f"   ✅ Login page: Accessible")
        
        print("\n🚀 Next Steps:")
        print("1. Test the signup functionality")
        print("2. Create a test account")
        print("3. Try creating a quiz")
        print("4. Add the URL to your resume!")
        
        print(f"\n🔗 Your live application: {app_url}")
        
    else:
        print("\n❌ Some issues were found with your deployment.")
        print("Please check the errors above and fix them.")
        print("\n💡 Common issues:")
        print("- Environment variables not set correctly")
        print("- Database connection issues")
        print("- API key problems")
        print("- Build/deployment errors")

if __name__ == "__main__":
    main()
