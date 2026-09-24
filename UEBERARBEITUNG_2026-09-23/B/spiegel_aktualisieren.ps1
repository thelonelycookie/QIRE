# Frischt das Buendel aus dem eigentlichen Projekt auf.
#
# Der Ordner hier ist ein Spiegel zum Lesen. Geaendert wird immer im Projekt;
# dieses Skript holt den Stand nach. Es kopiert NUR in diese Richtung und
# ueberschreibt den Spiegel vollstaendig -- lokale Aenderungen hier gehen
# verloren, und das ist Absicht.

$ErrorActionPreference = "Stop"

$Projekt = "C:\Users\Malte\Downloads\qire_ai_console_v1_0_remote_research_platform\qire_ai_console_v1_0"
$Ziel    = $PSScriptRoot

if (-not (Test-Path $Projekt)) {
    Write-Host "Projektordner nicht gefunden: $Projekt" -ForegroundColor Red
    Write-Host "Pfad in diesem Skript anpassen." -ForegroundColor Red
    exit 1
}

# --- Module der B-Serie ---
$Module = @(
    "result_store.py", "equivalence.py", "audit.py", "results_report.py",
    "figures.py", "spectral_probe.py", "structured_layers.py", "local_net.py",
    "validate_structured.py", "ablation_blockdiag.py", "ablation_locality.py",
    "ablation_structure.py", "ablation_orthogonality.py", "ablation_unitarity.py",
    "weight_structure.py", "rank_prediction.py", "sample_efficiency.py",
    "optimization_gap.py", "trajectory_dose.py", "rank_curriculum.py",
    "intrinsic_dimension.py", "tensor_train.py", "dmrg_adaptive_rank.py"
)
$Runner = @("run_project_b.py", "run_project_b2.py", "run_project_b4.py")

New-Item -ItemType Directory -Force -Path "$Ziel\code", "$Ziel\daten", "$Ziel\abbildungen" | Out-Null

$fehlend = @()
foreach ($m in $Module) {
    $q = Join-Path "$Projekt\qire_ai" $m
    if (Test-Path $q) { Copy-Item $q "$Ziel\code\" -Force } else { $fehlend += $m }
}
foreach ($r in $Runner) {
    $q = Join-Path $Projekt $r
    if (Test-Path $q) { Copy-Item $q "$Ziel\code\" -Force } else { $fehlend += $r }
}

# --- Rohdaten (nur B-Serie) ---
Get-ChildItem "$Projekt\results\experiments" -Filter "B-*" -File |
    Copy-Item -Destination "$Ziel\daten\" -Force
foreach ($b in @("equivalence_report.txt", "audit.txt")) {
    $q = Join-Path "$Projekt\results\experiments" $b
    if (Test-Path $q) { Copy-Item $q "$Ziel\daten\" -Force }
}

# --- Abbildungen der B-Serie ---
Get-ChildItem "$Projekt\results\figures" -Filter "abb*.png" -File -ErrorAction SilentlyContinue |
    Copy-Item -Destination "$Ziel\abbildungen\" -Force

# --- erzeugte Berichte ---
if (Test-Path "$Projekt\RESULTS.md") { Copy-Item "$Projekt\RESULTS.md" "$Ziel\" -Force }

Write-Host ""
Write-Host "Spiegel aufgefrischt aus:" -ForegroundColor Green
Write-Host "  $Projekt"
Write-Host ("  code        : {0} Dateien" -f (Get-ChildItem "$Ziel\code" -File).Count)
Write-Host ("  daten       : {0} Dateien" -f (Get-ChildItem "$Ziel\daten" -File).Count)
Write-Host ("  abbildungen : {0} Dateien" -f (Get-ChildItem "$Ziel\abbildungen" -File).Count)
if ($fehlend.Count -gt 0) {
    Write-Host ""
    Write-Host "Nicht gefunden (im Projekt umbenannt oder geloescht?):" -ForegroundColor Yellow
    $fehlend | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
}
Write-Host ""
Write-Host "TAGEBUCH_B.md wird NICHT automatisch aufgefrischt -- der Auszug" -ForegroundColor DarkGray
Write-Host "haengt an Zeilennummern im Tagebuch und muss von Hand nachgezogen" -ForegroundColor DarkGray
Write-Host "werden, wenn dort neue B-Eintraege dazukommen." -ForegroundColor DarkGray
