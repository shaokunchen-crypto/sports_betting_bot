@echo off
rem LIVE: this laptop can send orders. Only ONE laptop per API key should ever run this - see README.
cd /d "%~dp0"
start "" http://localhost:8003
.venv\Scripts\python.exe tdesk_server_v2.py --trade
