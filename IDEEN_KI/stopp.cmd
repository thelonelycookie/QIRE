@echo off
REM Sanfter Stopp: legt die Datei STOP an; die KI beendet sich nach dem aktuellen Schritt (< 5 min).
REM Notfall: stopp.cmd /hart  -> beendet sofort alle Prozesse (letzter Schritt kann verloren gehen,
REM gespeicherte Ideen in daten\ideen.jsonl bleiben erhalten).
cd /d "%~dp0"
if /i "%1"=="/hart" goto hart
echo stop> STOP
echo STOP-Datei angelegt. Die KI beendet sich nach dem laufenden Schritt (hoechstens wenige Minuten).
goto ende
:hart
echo stop> STOP
if exist daten\betrieb.pid (
  for /f %%p in (daten\betrieb.pid) do taskkill /F /T /PID %%p
)
taskkill /F /IM llama-server.exe
echo Alles hart beendet.
:ende
