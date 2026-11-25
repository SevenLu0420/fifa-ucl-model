#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-match}"

if [[ "$MODE" == "full" ]]; then
  echo '[1/6] Integrate FIFA15-FC27 raw files -> data/output/'
  (cd data && python build_dataset.py --root .)
  echo '[2/6] Check integrated outputs'
  (cd data && python check_outputs.py)
  echo '[3/6] Finalize player/club/coach tables -> data/output_final/'
  (cd data && python finalize_dataset.py --input output --out output_final --history-teams raw/history/male_teams.csv --fc24-zip raw/fc24/24.zip --eafc-zip raw/eafc/eafc_db_output.zip)
  echo '[4/6] Audit UCL club coverage -> data/coverage_audit/'
  (cd data && python audit_ucl_coverage.py --input output_final --out coverage_audit --roster-csv reference/historical_ucl_teams_2014_2024.csv --known-map reference/ucl_25_27_verified_mapping.csv --history-teams raw/history/male_teams.csv --fc24-zip raw/fc24/24.zip --eafc-zip raw/eafc/eafc_db_output.zip --offline)
  echo '[5/6] FIFA-level exploratory analysis -> data/eda_output/'
  (cd data && python run_eda.py --data output_final --coverage coverage_audit/ucl_teams_coverage.csv --out eda_output)
  echo '[6/6] Build match datasets'
elif [[ "$MODE" == "match" ]]; then
  echo '[1/1] Use shared FIFA final tables to build match datasets'
else
  echo 'Usage: bash run_pipeline.sh [match|full]' >&2
  exit 2
fi
bash run_match_builder.sh
printf '\n[DONE] Created data/match_dataset/. Next run the Notebook from repo root.\n'
