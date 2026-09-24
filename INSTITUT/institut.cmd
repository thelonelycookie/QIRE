@echo off
rem Startet das Institut mit Maltes QIRE-Umgebung. Beispiele:
rem   institut.cmd status
rem   institut.cmd vortest
rem   institut.cmd alle        (rechnet alle freigeschalteten Punkte, z. B. ueber Nacht)
cd /d "%~dp0"
set INSTITUT_PROZESSE=5
"C:\Users\Malte\Downloads\qire_ai_console_v1_0_remote_research_platform\qire_ai_console_v1_0\.venv\Scripts\python.exe" direktor.py %* >> ergebnisse\direktor.log 2>&1
type STAND.md
