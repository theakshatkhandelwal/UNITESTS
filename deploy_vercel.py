#!/usr/bin/env python3
"""
UniTest Vercel Deployment Helper Script
This script helps prepare your application for Vercel deployment
"""

import os
import sys
import subprocess
import json

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'vercel.json',
        'api/index.py',
        'requirements.txt',
        'app.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files present")
    return True

def check_env_vars():
    """Check if environment variables are set"""
    required_vars = [
        'SECRET_KEY',
        'GOOGLE_AI_API_KEY',
        'DATABASE_URL'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("⚠️  Missing environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nMake sure to set these in Vercel dashboard:")
        print("   - Go to your project settings")
        print("   - Navigate to Environment Variables")
        print("   - Add the missing variables")
        return False
    
    print("✅ All environment variables set")
    return True

def validate_vercel_config():
    """Validate vercel.json configuration"""
    try:
        with open('vercel.json', 'r') as f:
            config = json.load(f)
        
        required_keys = ['version', 'builds', 'routes', 'env']
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing key '{key}' in vercel.json")
                return False
        
        print("✅ Vercel configuration valid")
        return True
    except Exception as e:
        print(f"❌ Error validating vercel.json: {e}")
        return False

def check_dependencies():
    """Check if all dependencies are in requirements.txt"""
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        
        critical_deps = [
            'Flask',
            'Flask-SQLAlchemy',
            'Flask-Login',
            'google-generativeai',
            'psycopg2-binary'
        ]
        
        missing_deps = []
        for dep in critical_deps:
            if dep not in requirements:
                missing_deps.append(dep)
        
        if missing_deps:
            print("❌ Missing critical dependencies:")
            for dep in missing_deps:
                print(f"   - {dep}")
            return False
        
        print("✅ All critical dependencies present")
        return True
    except Exception as e:
        print(f"❌ Error checking dependencies: {e}")
        return False

def main():
    """Main deployment check function"""
    print("🚀 UniTest Vercel Deployment Check")
    print("=" * 40)
    
    checks = [
        ("File Structure", check_requirements),
        ("Vercel Configuration", validate_vercel_config),
        ("Dependencies", check_dependencies),
        ("Environment Variables", check_env_vars)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}:")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All checks passed! Your app is ready for Vercel deployment.")
        print("\nNext steps:")
        print("1. Push your code to GitHub")
        print("2. Connect your repository to Vercel")
        print("3. Set environment variables in Vercel dashboard")
        print("4. Deploy!")
        print("\n📖 See VERCEL_DEPLOYMENT_GUIDE.md for detailed instructions")
    else:
        print("❌ Some checks failed. Please fix the issues above before deploying.")
        sys.exit(1)

if __name__ == "__main__":
    main()
