# FIFA–UCL Project: Team Data & Reproduction Guide

Jiaxuan Liu

## 1. What to share

**Recommended (start modeling without rebuilding FIFA raw data):**

| Folder / files | Purpose | Share? |
|---|---|---|
| `data/output_final/players_2015_2027.csv` | Harmonized player attributes for FIFA15–FC27 | **Required** for rebuilding match features |
| `data/output_final/clubs_2015_2027.csv` | Club identities and overall/attack/midfield/defence ratings | **Required** for rebuilding match features |
| `data/output_final/club_features_2015_2027.csv` | Precomputed season-level squad/club aggregates | Recommended for research, not required by current match-builder command |
| `data/output_final/coaches_2015_2027.csv` | Coach data for future experiments | Optional |
| `data/coverage_audit/ucl_teams_coverage.csv` | Verified season–club IDs / coverage | **Required** for match builder |
| `data/reference/historical_ucl_teams_2014_2024.csv`, `ucl_25_27_verified_mapping.csv` | Source/reference club mappings | **Required** for full reproduction / match builder's known map |
| `data/raw/matches/resultats_ldc_par_saison.xlsx` | UCL results, 2015–16 to 2024–25 | **Required** for rebuilding matches |
| `data/reference/ucl_2014_15_matches.csv`, `ucl_2025_26_matches.csv`, `ucl_2026_27_league_schedule.csv`, `ucl_extra_time_90min_verified.csv` | Extra seasons, 2026–27 fixtures, verified 90-minute corrections | **Required** for rebuilding matches |
| `data/reference/ucl_2014_2027_supplement.xlsx` | Human-readable supporting workbook; CSVs above are used by script | Optional |
| `data/match_dataset/matches_compact.csv`, `matches_full.csv` | Historical match-level features, 93 / 797 original predictors | **Required** to run notebook directly |
| `data/match_dataset/matches_canonical.csv` | Audited historical + fixture match table | **Required** for notebook EDA |
| `data/match_dataset/fixtures_2026_27_compact.csv`, `fixtures_2026_27_full.csv` | 2026–27 fixture-level features | **Required** for notebook export/simulation prep |
| `data/match_dataset/compact_feature_columns.json`, `full_feature_columns.json`, `sample_status.csv`, `team_feature_bank.csv`, `team_name_mapping.csv`, `team_feature_quality.csv`, `build_report.json` (if present) | Schema, audit and diagnostics | Recommended |

**Optional large original inputs:** `data/raw/history/`, `data/raw/fc24/24.zip`, `data/raw/eafc/eafc_db_output.zip`, and the other FIFA/FC inputs specified in the main README. These are only needed to rerun **all** FIFA preprocessing. Do not redistribute third-party raw datasets without checking their licenses.

**Not needed to share:** `data/output/` (intermediate data), generated figures, saved models, or `analysis/eda_output/prepared/` unless teammates want to skip notebook preprocessing. These can be regenerated.

## 2. Setup

Clone the repo, put the shared files at the exact paths above, open a terminal **at the repository root**, and install dependencies:

```bash
python -m pip install pandas numpy openpyxl scipy matplotlib scikit-learn joblib notebook nbconvert xgboost catboost shap
```

## 3. Pipeline

| Step | Command (from repo root) | What it does | Output |
|---|---|---|---|
| 1 | `bash run_pipeline.sh full` | Integrates raw FIFA15–FC27 files, validates, finalizes and audits club coverage, then builds matches | `data/output/`, `data/output_final/`, `data/coverage_audit/`, `data/match_dataset/` |
| 1 | `bash run_pipeline.sh match` | Uses shared final FIFA tables and match records to build match datasets; skips raw-data integration | `data/match_dataset/` |
| 2 | Open the main project `.ipynb` from **repo root** and run Part I | Checks missing values, creates season-based train/valid/test splits, imputes training-only, exports model inputs | `analysis/eda_output/prepared/` |
| 3 | Run Part II of notebook | Match EDA and interpretable plots | `analysis/eda_output/figures/`, `tables/` |
| 4 | Run Part III of notebook | Trains six predictive tasks and compares models | `model/results/` |
| 5 | Run appended Part IV–V notebook cells, if present | Unified W/D/L evaluation; draw diagnosis and differential-feature experiments | `model/results/unified_wdl/`, `model/results/diff_diagnostics/` |

**The prepared `data/match_dataset/` folder is already shared, you can skip Step 1 entirely.**

## 4. Data split and current limitations

- Train: 2014–15 to 2023–24; Validation: 2024–25; Test: 2025–26. No random match split.
- Current historical matched sample: around 1.4k matches (exact number can change after mapping/venue fixes). Main compact/full builder uses 93/797 initial predictors.
- Detailed Full features contain 22 individual player slots; Compact includes squad aggregates. Missing values and normalization are handled within training-only preprocessing.
- Draw prediction is weak; high-dimensional Full features often overfit. Club-only ratings remain a strong baseline.
- Sabah and Slovan Bratislava have no verified FC27 club data in the supplied game database; do not fabricate club IDs. Simulation is not yet complete.

## 5. Collaboration

Keep the shared input tables unchanged. Make your own copy of the project notebook with a distinct filename (e.g. `model_yourname.ipynb`), explore alternative features/models, and share useful results and code changes with the group. Do not overwrite teammates' notebooks or original datasets.
