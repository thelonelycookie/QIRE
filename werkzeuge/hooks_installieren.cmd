@echo off
rem Aktiviert die versionierten Hooks aus werkzeuge\hooks (einmal pro Klon ausfuehren).
rem Setzt core.hooksPath auf werkzeuge/hooks (lokale Git-Einstellung, nicht Teil des Repos).
for /f "delims=" %%i in ('git rev-parse --show-toplevel') do cd /d "%%i"
git config core.hooksPath werkzeuge/hooks
if errorlevel 1 (echo Fehler: git config fehlgeschlagen & exit /b 1)
echo Hooks aktiv: core.hooksPath = werkzeuge/hooks
echo   pre-commit: Vorregistrierung (Modul+Kriterien der ID muessen frueher committet sein),
echo               Rauchtest-Ergebnisse nur auf Tuning-/Entwurfs-Seeds.
echo Umgehen im Ausnahmefall:  git commit --no-verify   (Grund in die Commit-Nachricht!)
echo Deaktivieren:             git config --unset core.hooksPath
