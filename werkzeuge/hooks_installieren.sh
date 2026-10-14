#!/bin/sh
# Aktiviert die versionierten Hooks aus werkzeuge/hooks (einmal pro Klon ausfuehren).
# Setzt core.hooksPath auf werkzeuge/hooks (lokale Git-Einstellung, nicht Teil des Repos).
set -e
cd "$(git rev-parse --show-toplevel)"
chmod +x werkzeuge/hooks/pre-commit 2>/dev/null || true
git config core.hooksPath werkzeuge/hooks
echo "Hooks aktiv: core.hooksPath = werkzeuge/hooks"
echo "  pre-commit: Vorregistrierung (Modul+Kriterien der ID muessen frueher committet sein),"
echo "              Rauchtest-Ergebnisse nur auf Tuning-/Entwurfs-Seeds."
echo "Umgehen im Ausnahmefall:  git commit --no-verify   (Grund in die Commit-Nachricht!)"
echo "Deaktivieren:             git config --unset core.hooksPath"
