@echo off
cd /d "%~dp0"
echo Avvio della Web App ECG in corso...
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
python -m streamlit run app.py
pause
