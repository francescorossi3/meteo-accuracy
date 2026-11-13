#!/usr/bin/env bash
# Prova completa con DATI SINTETICI (nessuna connessione necessaria per i dati).
set -e
cd "$(dirname "$0")"
export METEOITA_DATA=demo_data
python scripts/demo_generate.py
python scripts/03_verify.py --from "$(date -u -d '-80 days' +%F 2>/dev/null || date -u -v-80d +%F)"
python scripts/04_train_gpr.py
python scripts/05_build_maps.py --demo
python scripts/06_build_site.py
echo "Apri http://localhost:8000  (Ctrl+C per chiudere)"
python -m http.server -d build/site 8000
