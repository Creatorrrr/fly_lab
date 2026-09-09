@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
 echo First: py -3.12 -m venv .venv
 echo Then: .venv\Scripts\python -m pip install -r requirements.txt
 exit /b 2
)
.venv\Scripts\python.exe run.py %*
