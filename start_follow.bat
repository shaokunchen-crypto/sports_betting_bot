@echo off
rem LIVE, SECOND DESK: fair = the mid of the pricing desk's quotes (TEAM_OWNERS); quotes on its OWN API key
rem (my_keys.py) only where the pricing desk quotes both sides. See README.
cd /d "%~dp0"
start "" http://localhost:8003
.venv\Scripts\python.exe tdesk_server.py --trade --follow --rate 9
