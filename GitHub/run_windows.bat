@echo off
cd /d "%~dp0"
python -m venv .venv
call .venv\Scriptsctivate
pip install -r requirements-tested.txt || pip install -r requirements.txt
python reproduce_article2.py
pause
