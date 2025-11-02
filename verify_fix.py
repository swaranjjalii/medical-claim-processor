"""
Quick verification script to test if the fix is working
Run this BEFORE testing the API to verify environment setup
"""

import os
import sys
from pathlib import Path

print("=" * 60)
print("🔍 Medical Claim Processor - Environment Verification")
print("=" * 60)
print()

# Color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
END = '\033[0m'

def check_mark(condition):
    return f"{GREEN}✅{END}" if condition else f"{RED}❌{END}"

def warning_mark():
    return f"{YELLOW}⚠️{END}"

# Check 1: Python version
print(f"{BLUE}1. Checking Python version...{END}")
python_version = sys.version_info
version_ok = python_version.major == 3 and python_version.minor >= 9
print(f"   {check_mark(version_ok)} Python {python_version.major}.{python_version.minor}.{python_version.micro}")
if not version_ok:
    print(f"   {RED}Error: Python 3.9+ required{END}")
print()

# Check 2: Dependencies
print(f"{BLUE}2. Checking required packages...{END}")
required_packages = {
    'fastapi': False,
    'uvicorn': False,
    'pydantic': False,
    'python-dotenv': False,
    'openai': False,
    'groq': False
}

for package in required_packages.keys():
    try:
        __import__(package.replace('-', '_'))
        required_packages[package] = True
        print(f"   {check_mark(True)} {package}")
    except ImportError:
        print(f"   {check_mark(False)} {package}")

all_packages_installed = all(required_packages.values())
if not all_packages_installed:
    print(f"\n   {RED}Missing packages! Install with:{END}")
    print(f"   pip install -r requirements.txt")
print()

# Check 3: .env file
print(f"{BLUE}3. Checking .env configuration...{END}")
env_exists = Path(".env").exists()
print(f"   {check_mark(env_exists)} .env file exists")

if env_exists:
    from dotenv import load_dotenv
    load_dotenv()
    
    openai_key = os.getenv("OPENAI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    has_openai = openai_key and openai_key != "your_openai_key_here" and openai_key != "sk-proj-your-key-here"
    has_groq = groq_key and groq_key != "your_groq_key_here" and groq_key != "gsk-your-key-here"
    
    print(f"   {check_mark(has_openai)} OpenAI API key configured")
    if has_openai:
        print(f"      Key: {openai_key[:20]}...{openai_key[-4:]}")
    
    print(f"   {check_mark(has_groq)} Groq API key configured")
    if has_groq:
        print(f"      Key: {groq_key[:20]}...{groq_key[-4:]}")
    
    if not has_openai and not has_groq:
        print(f"\n   {RED}Error: No valid API key found!{END}")
        print(f"   Please update .env with either:")
        print(f"   - OPENAI_API_KEY=sk-proj-...")
        print(f"   - GROQ_API_KEY=gsk_...")
    elif has_openai:
        print(f"\n   {GREEN}✅ Will use OpenAI (preferred){END}")
    else:
        print(f"\n   {GREEN}✅ Will use Groq{END}")
else:
    print(f"\n   {RED}Error: .env file not found!{END}")
    print(f"   Create .env with your API key:")
    print(f"   echo 'OPENAI_API_KEY=sk-proj-your-key' > .env")
print()

# Check 4: main.py exists
print(f"{BLUE}4. Checking application files...{END}")
main_exists = Path("main.py").exists()
print(f"   {check_mark(main_exists)} main.py")

if main_exists:
    # Try to import and check
    try:
        import main
        print(f"   {GREEN}✅ main.py can be imported{END}")
        
        # Check if key classes exist
        has_llm_client = hasattr(main, 'LLMClient')
        has_orchestrator = hasattr(main, 'ClaimProcessingOrchestrator')
        has_app = hasattr(main, 'app')
        
        print(f"   {check_mark(has_llm_client)} LLMClient class")
        print(f"   {check_mark(has_orchestrator)} ClaimProcessingOrchestrator class")
        print(f"   {check_mark(has_app)} FastAPI app")
        
    except Exception as e:
        print(f"   {RED}❌ Error importing main.py: {e}{END}")
print()

# Check 5: Test LLM initialization
print(f"{BLUE}5. Testing LLM client initialization...{END}")
if env_exists and all_packages_installed and main_exists:
    try:
        from main import LLMClient
        llm = LLMClient()
        print(f"   {GREEN}✅ LLM client initialized successfully{END}")
        print(f"   Provider: {llm.provider}")
        print(f"   Model: {llm.model}")
    except Exception as e:
        print(f"   {RED}❌ Failed to initialize LLM client: {e}{END}")
print()

# Summary
print("=" * 60)
print("📊 SUMMARY")
print("=" * 60)

all_ok = (
    version_ok and 
    all_packages_installed and 
    env_exists and 
    main_exists and
    (has_openai or has_groq if env_exists else False)
)

if all_ok:
    print(f"{GREEN}✅ All checks passed! You're ready to go!{END}")
    print()
    print("🚀 Next steps:")
    print("   1. Start the server:")
    print(f"      {BLUE}uvicorn main:app --reload{END}")
    print()
    print("   2. Test the API:")
    print(f"      {BLUE}python test_client.py{END}")
    print()
    print("   3. Or visit the docs:")
    print(f"      {BLUE}http://localhost:8000/docs{END}")
else:
    print(f"{RED}❌ Some checks failed. Please fix the issues above.{END}")
    print()
    print("📝 Quick fixes:")
    if not version_ok:
        print("   - Install Python 3.9 or higher")
    if not all_packages_installed:
        print("   - Run: pip install -r requirements.txt")
    if not env_exists or not (has_openai or has_groq):
        print("   - Create .env file with valid API key")
        print("     OpenAI: https://platform.openai.com/api-keys")
        print("     Groq: https://console.groq.com")

print("=" * 60)