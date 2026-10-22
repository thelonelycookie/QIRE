#!/bin/bash
# Einmal nach dem ersten Login auf dem MPCDF ausfuehren (Login-Knoten). Gibt aus, was in lokal.env gehoert.
#   bash MPI/pruefe_cluster.sh | tee logs/cluster_info.txt
echo "== System: $(hostname)  $(date +%F)"
echo "== Partitionen und Zeitlimits (Spalte TIMELIMIT = Maximum je Job)"
sinfo -o "%P %l %D %c %m" 2>/dev/null || echo "sinfo fehlt"
echo "== Eigene Accounts/QOS"
sacctmgr -n show assoc user="$USER" format=account,partition,qos,maxwall 2>/dev/null
echo "== Python-Module"
module avail python 2>&1 | head -20; module avail anaconda 2>&1 | head -5
echo "== Speicher (Home, /ptmp)"
quota -s 2>/dev/null | head; df -h "$HOME" /ptmp/"$USER" 2>/dev/null
echo "== Internet vom Login-Knoten"; timeout 5 curl -sI https://pypi.org >/dev/null && echo ja || echo nein
