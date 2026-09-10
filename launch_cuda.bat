@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe -X utf8 run_c.py --backend exp_lif_cuda %*
