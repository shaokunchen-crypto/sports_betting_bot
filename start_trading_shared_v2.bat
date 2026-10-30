@echo off
rem LIVE and SHARED: this laptop runs the one desk that trades; up to a few teammates open the link it
rem prints (same network) and use the same desk - same quotes, results, edits. See README.
cd /d "%~dp0"
start "" http://localhost:8003
.venv\Scripts\python.exe tdesk_server_v2.py --trade --share
