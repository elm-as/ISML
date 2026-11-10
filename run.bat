@echo off
setlocal
cd /d "%~dp0"
echo ========================================================
echo ISML: Impact of Social Media on Life (4,500 Cohort)
echo ========================================================
"C:\Users\elmas\AppData\Local\Programs\Python\Python310\python.exe" main.py %*
pause

