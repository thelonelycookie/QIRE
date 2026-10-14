@echo off
rem Routine-Worker (Windows): Python ueber A\py.cmd, aus dem Ordner A heraus (WERKSTATT\CLAUDE.md, Regel 3).
rem Beispiel:  WERKSTATT\routine\routine.cmd bericht
pushd "%~dp0..\..\A"
call py.cmd "%~dp0routine.py" %*
set RC=%ERRORLEVEL%
popd
exit /b %RC%
