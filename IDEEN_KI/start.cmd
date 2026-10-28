@echo off
REM Startet die lokale QIRE-Ideen-KI (Dauerbetrieb). Stoppen: stopp.cmd (oder Datei STOP anlegen).
cd /d "%~dp0"
if exist STOP del STOP
if not exist daten mkdir daten
echo Ideen-KI startet ... (Stopp: stopp.cmd)
python -m schmiede.betrieb %* >> daten\betrieb.log 2>&1
echo Ideen-KI beendet.
