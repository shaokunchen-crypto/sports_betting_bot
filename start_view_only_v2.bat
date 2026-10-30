@echo off
rem View only: prices, board, positions, P&L. Cannot send or cancel orders.
cd /d "%~dp0"
start "" http://localhost:8003
.venv\Scripts\python.exe tdesk_server_v2.py
