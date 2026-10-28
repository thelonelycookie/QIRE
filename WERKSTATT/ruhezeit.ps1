# Abends per Aufgabenplanung: Worker beenden und Modell aus dem VRAM nehmen.
# Laufende Simulationen (python) werden NICHT beendet, damit keine Daten verloren gehen;
# monitor.py meldet, wenn sie fertig sind.
Get-Process | Where-Object { $_.MainWindowTitle -like "werkstatt-*" } | Stop-Process -Force
foreach ($m in (ollama ps | Select-Object -Skip 1 | ForEach-Object { ($_ -split '\s+')[0] })) {
    if ($m) { ollama stop $m }
}
Write-Host "Ruhezeit: Worker beendet, Modell entladen."
