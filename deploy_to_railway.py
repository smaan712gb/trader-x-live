#!/usr/bin/env python3
"""
Railway Deployment Helper for Trader-X
Automates the deployment process to Railway
"""

import os
import subprocess
import sys
from pathlib import Path

def check_requirements():
    """Check if all required files exist for deployment"""
    required_files = [
        'web_dashboard.py',
        'requirements_railway.txt',
        'core/ai_engine.py',
        'technical_analysis_engine.py',
        'data/fmp_market_data.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files present")
    return True

def check_environment_variables():
    """Check if required environment variables are documented"""
    required_vars = [
        'XAI_API_KEY',
        'FMP_API_KEY', 
        'SUPABASE_URL',
        'SUPABASE_KEY'
    ]
    
    print("\n📋 Required Environment Variables for Railway:")
    print("   Set these in Railway Dashboard → Variables tab:")
    print()
    
    for var in required_vars:
        value = os.getenv(var, 'NOT_SET')
        status = "✅" if value != 'NOT_SET' else "❌"
        print(f"   {status} {var}={value[:10]}..." if value != 'NOT_SET' else f"   {status} {var}=NOT_SET")
    
    print()
    print("   Additional Railway Variables:")
    print("   ✅ PORT=8501 (Railway sets automatically)")
    print("   ✅ PYTHONPATH=/app")
    print("   ✅ STREAMLIT_SERVER_HEADLESS=true")
    print()

def create_railway_files():
    """Create Railway-specific configuration files"""
    
    # Create Procfile for Railway
    procfile_content = "web: streamlit run web_dashboard.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true\n"
    
    with open('Procfile', 'w') as f:
        f.write(procfile_content)
    print("✅ Created Procfile")
    
    # Create .railwayignore
    railwayignore_content = """
# Railway Ignore File
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git/
.mypy_cache/
.pytest_cache/
.hypothesis/
.DS_Store
*.sqlite
*.db
.env.local
.env.development
.env.test
.env.production
trader_x_*.log
test_*.py
demo_*.py
"""
    
    with open('.railwayignore', 'w') as f:
        f.write(railwayignore_content)
    print("✅ Created .railwayignore")

def test_local_deployment():
    """Test the application locally before deployment"""
    print("\n🧪 Testing local deployment...")
    
    try:
        # Test if streamlit can import the main app
        result = subprocess.run([
            sys.executable, '-c', 
            'import web_dashboard; print("✅ web_dashboard imports successfully")'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Application imports successfully")
        else:
            print("❌ Import error:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ Import test timed out")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

def show_deployment_instructions():
    """Show step-by-step deployment instructions"""
    print("\n🚀 Railway Deployment Instructions:")
    print("=" * 50)
    
    print("\n1. 📁 Prepare Files:")
    print("   ✅ All files ready for deployment")
    
    print("\n2. 🌐 Railway Dashboard:")
    print("   • Go to: https://railway.app/dashboard")
    print("   • Click 'New Project'")
    print("   • Select 'Deploy from GitHub repo' or 'Empty Service'")
    print("   • Name: 'trader-x-live'")
    
    print("\n3. ⚙️ Environment Variables:")
    print("   • Go to Variables tab in Railway")
    print("   • Add the variables shown above")
    print("   • Make sure no trailing spaces in values")
    
    print("\n4. 🔧 Deploy Settings:")
    print("   • Start Command: streamlit run web_dashboard.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true")
    print("   • Root Directory: / (default)")
    print("   • Build Command: pip install -r requirements_railway.txt")
    
    print("\n5. 🚀 Deploy:")
    print("   • Click 'Deploy' button")
    print("   • Wait 2-3 minutes for build")
    print("   • Get Railway URL (format: https://trader-x-live-production.up.railway.app)")
    
    print("\n6. ✅ Test Deployment:")
    print("   • Open Railway URL in browser")
    print("   • Test with symbol: NVDA")
    print("   • Verify Grok-4 Live Search works")
    print("   • Share URL with team")
    
    print("\n💰 Expected Costs:")
    print("   • Railway: $5-20/month")
    print("   • Live Search: $0.025/source")
    print("   • Team usage: $50-300/month")
    
    print("\n📞 Support:")
    print("   • Railway Docs: https://docs.railway.app")
    print("   • Railway Discord: Railway community")
    print("   • Deployment Guide: RAILWAY_DEPLOYMENT_GUIDE.md")

def main():
    """Main deployment helper function"""
    print("🚀 Trader-X Railway Deployment Helper")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Deployment requirements not met")
        return False
    
    # Check environment variables
    check_environment_variables()
    
    # Create Railway files
    print("📁 Creating Railway configuration files...")
    create_railway_files()
    
    # Test local deployment
    if not test_local_deployment():
        print("\n⚠️ Local tests failed - proceed with caution")
    
    # Show deployment instructions
    show_deployment_instructions()
    
    print("\n✅ Ready for Railway deployment!")
    print("📖 See RAILWAY_DEPLOYMENT_GUIDE.md for detailed instructions")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
