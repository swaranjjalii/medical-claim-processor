@echo off
REM Medical Claim Processor - Windows Setup Script

echo.
echo ===================================================
echo Medical Insurance Claim Processor - Setup Script
echo ===================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed!
    echo Please install Python 3.9 or higher from python.org
    pause
    exit /b 1
)

echo [+] Python found
echo.

REM Create project structure
echo [+] Creating project structure...
if not exist .vscode mkdir .vscode
if not exist test_files mkdir test_files
if not exist logs mkdir logs
echo [+] Directories created
echo.

REM Create virtual environment
echo [+] Creating virtual environment...
if not exist venv (
    python -m venv venv
    echo [+] Virtual environment created
) else (
    echo [!] Virtual environment already exists
)
echo.

REM Activate virtual environment
echo [+] Activating virtual environment...
call venv\Scripts\activate.bat
echo [+] Virtual environment activated
echo.

REM Upgrade pip
echo [+] Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo [+] pip upgraded
echo.

REM Install dependencies
echo [+] Installing dependencies...
if exist requirements.txt (
    pip install -r requirements.txt
    echo [+] Dependencies installed
) else (
    echo [!] requirements.txt not found. Skipping...
)
echo.

REM Create .env file
if not exist .env (
    echo [+] Creating .env file...
    (
        echo # Groq API Configuration
        echo GROQ_API_KEY=your_groq_api_key_here
        echo.
        echo # Model Configuration
        echo MODEL_NAME=mixtral-8x7b-32768
        echo.
        echo # Server Configuration
        echo HOST=0.0.0.0
        echo PORT=8000
        echo LOG_LEVEL=INFO
        echo.
        echo # File Processing Limits
        echo MAX_FILE_SIZE_MB=10
        echo MAX_FILES_PER_REQUEST=10
    ) > .env
    echo [+] .env file created
    echo [!] Please update .env with your Groq API key!
) else (
    echo [!] .env file already exists
)
echo.

REM Create VS Code settings
echo [+] Creating VS Code configuration...
(
    echo {
    echo   "python.defaultInterpreterPath": "${workspaceFolder}/venv/Scripts/python.exe",
    echo   "python.linting.enabled": true,
    echo   "python.linting.flake8Enabled": true,
    echo   "python.formatting.provider": "black",
    echo   "editor.formatOnSave": true,
    echo   "editor.rulers": [100],
    echo   "files.exclude": {
    echo     "**/__pycache__": true,
    echo     "**/*.pyc": true,
    echo     "**/venv": true
    echo   }
    echo }
) > .vscode\settings.json
echo [+] VS Code settings created
echo.

REM Create launch.json
(
    echo {
    echo   "version": "0.2.0",
    echo   "configurations": [
    echo     {
    echo       "name": "FastAPI: Run Server",
    echo       "type": "python",
    echo       "request": "launch",
    echo       "module": "uvicorn",
    echo       "args": ["main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
    echo       "jinja": true,
    echo       "justMyCode": true
    echo     }
    echo   ]
    echo }
) > .vscode\launch.json
echo [+] Launch configuration created
echo.

REM Create .gitignore
echo [+] Creating .gitignore...
(
    echo # Python
    echo __pycache__/
    echo *.py[cod]
    echo venv/
    echo .venv
    echo.
    echo # Environment
    echo .env
    echo.
    echo # IDE
    echo .vscode/
    echo .idea/
    echo.
    echo # Logs
    echo logs/
    echo *.log
    echo.
    echo # Test files
    echo test_files/
    echo *.pdf
    echo.
    echo # OS
    echo .DS_Store
    echo Thumbs.db
) > .gitignore
echo [+] .gitignore created
echo.

REM Create sample test file
echo [+] Creating sample test file...
(
    echo This is a sample medical bill file for testing purposes.
    echo Hospital: City General Hospital
    echo Amount: $15,000.00
    echo Date: 2024-10-15
) > test_files\sample_bill.txt
echo [+] Sample test file created
echo.

REM Final instructions
echo.
echo ===================================================
echo Setup completed successfully!
echo ===================================================
echo.
echo Next steps:
echo.
echo 1. Update your Groq API key in .env file:
echo    notepad .env
echo.
echo 2. Start the development server:
echo    uvicorn main:app --reload
echo.
echo 3. Open your browser and visit:
echo    http://localhost:8000/docs
echo.
echo 4. Or press F5 in VS Code to start debugging
echo.
echo 5. Test the API with:
echo    python test_client.py
echo.
echo Don't forget to get your free Groq API key at:
echo https://console.groq.com
echo.
echo Happy coding!
echo.
pause