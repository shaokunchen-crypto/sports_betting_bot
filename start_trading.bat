@echo off
rem LIVE: this laptop can send orders. Each desk needs its OWN API key (my_keys.py) - see README.
cd /d "%~dp0"
start "" http://localhost:8003
.venv\Scripts\python.exe tdesk_server.py --trade --rate 9
