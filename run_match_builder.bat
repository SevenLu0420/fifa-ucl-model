@echo off
setlocal

REM Run from the repository root, even when launched from elsewhere.
cd /d "%~dp0"

python data\build_match_dataset.py ^
  --matches data\raw\matches\resultats_ldc_par_saison.xlsx ^
  --historical-csv data\reference\ucl_2014_15_matches.csv ^
  --historical-csv data\reference\ucl_2025_26_matches.csv ^
  --future-schedule data\reference\ucl_2026_27_league_schedule.csv ^
  --corrections data\reference\ucl_extra_time_90min_verified.csv ^
  --players data\output_final\players_2015_2027.csv ^
  --clubs data\output_final\clubs_2015_2027.csv ^
  --coverage data\coverage_audit\ucl_teams_coverage.csv ^
  --known-map data\reference\ucl_25_27_verified_mapping.csv ^
  --out data\match_dataset

if errorlevel 1 (
  echo [ERROR] Match dataset build failed.
  exit /b 1
)
echo [OK] Match dataset build completed.
endlocal
