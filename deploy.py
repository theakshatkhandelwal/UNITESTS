#!/usr/bin/env python3
"""
UniTest Deployment Helper Script
This script helps automate the deployment process for the UniTest application.
"""

import os
import subprocess
import sys
import json
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def check_prerequisites():
    """Check if required tools are installed."""
    print("🔍 Checking prerequisites...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    print("✅ requirements.txt found")
    
    # Check if app.py exists
    if not Path("app.py").exists():
        print("❌ app.py not found")
        return False
    
    print("✅ app.py found")
    
    return True

def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Create virtual environment if it doesn't exist
    if not Path("venv").exists():
        print("Creating virtual environment...")
        run_command("python -m venv venv", "Creating virtual environment")
    
    # Activate virtual environment and install dependencies
    if os.name == 'nt':  # Windows
        pip_command = "venv\\Scripts\\pip install -r requirements.txt"
    else:  # Unix/Linux/Mac
        pip_command = "venv/bin/pip install -r requirements.txt"
    
    return run_command(pip_command, "Installing dependencies")

def create_env_file():
    """Create .env file with required environment variables."""
    print("\n🔐 Creating environment file...")
    
    env_content = """# Flask Configuration
SECRET_KEY=your-secret-key-here-change-this-in-production
FLASK_ENV=development

# Database Configuration
DATABASE_URL=sqlite:///unittest.db

# Google AI API Key
GOOGLE_AI_API_KEY=your-google-ai-api-key-here

# Instructions:
# 1. Get your Google AI API key from: https://makersuite.google.com/app/apikey
# 2. Change the SECRET_KEY to a random string
# 3. For production, use a proper database URL
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ .env file created")
    print("⚠️  Please update the .env file with your actual API keys and secret key")

def test_local_run():
    """Test if the application runs locally."""
    print("\n🧪 Testing local run...")
    
    # Check if the app can be imported
    try:
        import app
        print("✅ Application imports successfully")
    except ImportError as e:
        print(f"❌ Application import failed: {e}")
        return False
    
    print("✅ Application is ready for local testing")
    print("💡 Run 'python app.py' to start the local server")
    return True

def setup_git():
    """Initialize git repository if not already done."""
    if not Path(".git").exists():
        print("\n📝 Initializing git repository...")
        run_command("git init", "Initializing git")
        run_command("git add .", "Adding files to git")
        run_command('git commit -m "Initial commit"', "Making initial commit")
        print("✅ Git repository initialized")
    else:
        print("✅ Git repository already exists")

def heroku_deployment_guide():
    """Show Heroku deployment guide."""
    print("\n🚀 Heroku Deployment Guide:")
    print("=" * 50)
    print("1. Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli")
    print("2. Login to Heroku: heroku login")
    print("3. Create app: heroku create your-unittest-app")
    print("4. Set environment variables:")
    print("   heroku config:set SECRET_KEY=your-secret-key")
    print("   heroku config:set GOOGLE_AI_API_KEY=your-api-key")
    print("5. Add PostgreSQL: heroku addons:create heroku-postgresql:mini")
    print("6. Deploy: git push heroku main")
    print("7. Open: heroku open")

def railway_deployment_guide():
    """Show Railway deployment guide."""
    print("\n🚂 Railway Deployment Guide:")
    print("=" * 50)
    print("1. Go to: https://railway.app/")
    print("2. Sign up with GitHub")
    print("3. Click 'New Project' > 'Deploy from GitHub repo'")
    print("4. Select your repository")
    print("5. Set environment variables in Railway dashboard:")
    print("   - SECRET_KEY")
    print("   - GOOGLE_AI_API_KEY")
    print("6. Deploy automatically on every push")

def render_deployment_guide():
    """Show Render deployment guide."""
    print("\n🎨 Render Deployment Guide:")
    print("=" * 50)
    print("1. Go to: https://render.com/")
    print("2. Sign up with GitHub")
    print("3. Click 'New' > 'Web Service'")
    print("4. Connect your GitHub repository")
    print("5. Set environment variables:")
    print("   - SECRET_KEY")
    print("   - GOOGLE_AI_API_KEY")
    print("6. Deploy")

def main():
    """Main deployment helper function."""
    print("🎓 UniTest Deployment Helper")
    print("=" * 40)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed. Please fix the issues above.")
        return
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Dependency installation failed.")
        return
    
    # Create environment file
    create_env_file()
    
    # Test local run
    if not test_local_run():
        print("\n❌ Local testing failed.")
        return
    
    # Setup git
    setup_git()
    
    # Show deployment guides
    print("\n" + "=" * 60)
    print("🎉 Your UniTest application is ready for deployment!")
    print("=" * 60)
    
    heroku_deployment_guide()
    railway_deployment_guide()
    render_deployment_guide()
    
    print("\n" + "=" * 60)
    print("📚 Next Steps:")
    print("1. Update your .env file with real API keys")
    print("2. Test locally: python app.py")
    print("3. Choose a deployment platform from above")
    print("4. Deploy and get your live URL!")
    print("5. Add the URL to your resume 🎯")
    print("=" * 60)

if __name__ == "__main__":
    main()
