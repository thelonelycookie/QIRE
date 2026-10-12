# Startet einen lokalen Worker: Claude Code mit Qwen ueber Ollama + Remote Control.
# Aufruf aus dem Projektordner:  .\WERKSTATT\start_worker.ps1 -Rolle code
param(
    [ValidateSet("code", "review", "daten")] [string]$Rolle = "code",
    [string]$Modell = "qwen3:8b",
    [int]$RuheAb = 0,
    [int]$RuheBis = 6,
    [switch]$MitToken   # nur falls ohne Token die Anmeldung scheitert
)

$stunde = (Get-Date).Hour
if ($RuheAb -lt $RuheBis) { $ruhe = ($stunde -ge $RuheAb -and $stunde -lt $RuheBis) }
else { $ruhe = ($stunde -ge $RuheAb -or $stunde -lt $RuheBis) }
if ($ruhe) {
    Write-Host "Ruhezeit ($RuheAb-$RuheBis Uhr): kein Start." -ForegroundColor Yellow
    exit 1
}

$projekt = Split-Path -Parent $PSScriptRoot
Set-Location $projekt

if (-not (ollama list | Select-String -SimpleMatch $Modell)) {
    Write-Host "Modell $Modell fehlt. Vorhanden:" -ForegroundColor Red
    ollama list
    exit 1
}

$env:ANTHROPIC_BASE_URL = "http://localhost:11434"
$env:ANTHROPIC_MODEL = $Modell
$env:ANTHROPIC_SMALL_FAST_MODEL = $Modell
if ($MitToken) { $env:ANTHROPIC_AUTH_TOKEN = "ollama" }

$regeln = Get-Content -Raw "$PSScriptRoot\CLAUDE.md"
$rolle = Get-Content -Raw "$PSScriptRoot\rollen\$Rolle.md"

$Host.UI.RawUI.WindowTitle = "werkstatt-$Rolle"
Write-Host "Starte werkstatt-$Rolle mit $Modell in $projekt" -ForegroundColor Green
Write-Host "Falls Remote Control nicht automatisch an ist: /remote-control eingeben." -ForegroundColor Cyan

claude --remote-control `
       --settings "$PSScriptRoot\worker-settings.json" `
       --append-system-prompt "$regeln`n`n$rolle`n`nDein Name: werkstatt-$Rolle"
