@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Starting the ML exercise app...
python -m streamlit run app.py
pause
