# FIFA/EA FC Data for UEFA Champions League Prediction (2014–2027)

**Columbia University — Data Analytics & Machine Learning, Group 17**

## Project overview

This repository prepares a consistent **men's player-, club-, and squad-level dataset for 13 FIFA/EA Sports FC editions (FIFA 15–FC 27)**. The intended downstream task is to predict UEFA Champions League match outcomes and simulate the 2026–27 competition. **Match results, match-level training examples, predictive models, and tournament simulation are not included yet.**

## Current progress

- Integrated historical SoFIFA-style player/team/coach records (FIFA 15–FC 24) with extracted EA game-database records (FC 25–FC 27).
- Harmonized player ratings, potential, detailed attributes, club ratings, contracts, squad membership, and available tactical information across editions. Kept attributes with no reliable historical equivalent as optional/missing rather than inventing values.
- Applied edition-based snapshot selection, limited evidence-based cross-season age completion, and additional data-quality checks. Player names are retained; unnecessary cosmetic attributes are excluded.
- Produced **234,068 player–club records** and **8,557 club–season records**, plus coach associations and club-level aggregate features (13 seasons).
- Audited men's UCL participants: **397 of 428 team–season entries matched (92.8%)**, with **31 unmatched entries** flagged for review. `NOT_FOUND` means absent from the *available data snapshots*, not necessarily absent from the entire FIFA/FC franchise.
- Completed basic EDA: attribute availability, ratings, team strength/depth, positions, tactics, and correlations. Note that the current “Top 11” is rating-ranked, **not an actual or position-constrained starting XI**.

## Data sources

| Seasons / purpose | Source | Usage |
| --- | --- | --- |
| FIFA 15–23 | [SoFIFA FIFA 23 complete historical dataset (Kaggle)](https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset) | Historical male players, clubs, coaches and updates |
| FC 24 | [EA Sports FC 24 complete player dataset (Kaggle)](https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset) | Historical male players, teams and coaches |
| FC 25–27 | [eafc-db-kit (GitHub)](https://github.com/ViniMacacari/eafc-db-kit) | Core native game tables: players, teams, squad links, leagues, managers |
| FC 25 | [nyagami FC25 ratings (Kaggle)](https://www.kaggle.com/datasets/nyagami/ea-sports-fc-25-database-ratings-and-stats) | Auxiliary player identities and summary ratings |
| FC 26–27 | [EA SPORTS FC Ratings](https://www.ea.com/games/ea-sports-fc/ratings) (retrieved JSON API data) | Supplementary attributes and six overall stat categories |
| UCL history / coverage | [OpenFootball Champions League (GitHub)](https://github.com/openfootball/champions-league), [UEFA](https://www.uefa.com/uefachampionsleague/) | Group/league-phase club lists and club-name reconciliation |

Additional candidate sources were reviewed but are **not required** for the reproduced core dataset, including [FC26 rovnez](https://www.kaggle.com/datasets/rovnez/fc-26-fifa-26-player-data), [FC25 SoFIFA merge](https://www.kaggle.com/datasets/sametozturkk/ea-sports-fc-25-real-player-data-sofifa-merge), and [FC27 mikedpad](https://www.kaggle.com/datasets/mikedpad/ea-sports-fc27-player-ratings). The original full CSVs for these optional sources are needed before their extra features can be incorporated.

**Why no live SoFIFA crawler/API?** We tested direct HTTP scraping and browser automation (including Selenium/Playwright), but SoFIFA access repeatedly returned HTTP 403/Cloudflare challenges. Its [official partner API](https://sofifa.com/document) requires authorization and a static IP allowlist; a test request also returned 403. We therefore use downloadable datasets and locally parsed EA game databases. EA's ratings API was usable for additional player attributes, but not complete career/team history. No request throttling or access-control bypass is attempted.

## Repository structure

```text
fifa15_fc27/
├── README.md
├── .gitignore
├── requirements.txt
├── build_dataset.py
├── check_outputs.py
├── finalize_dataset.py
├── audit_ucl_coverage.py
├── run_eda.py
├── field_dictionary.csv
├── FINAL_COLUMNS.md
├── reference/
│   ├── historical_ucl_teams_2014_2024.csv
│   └── ucl_25_27_verified_mapping.csv
├── raw/                         # local input files; NOT committed to Git
│   ├── history/
│   │   ├── male_players.csv
│   │   ├── male_teams.csv
│   │   └── male_coaches.csv
│   ├── fc24/24.zip
│   ├── eafc/
│   │   ├── eafc_db_output.zip
│   │   └── eafc_core_cleaned.zip
│   ├── fc25/all_players.csv
│   ├── api/ea_fc_data.zip
│   └── optional/              # optional full supplemental CSVs
├── output/                      # generated, local
├── output_final/                # generated, local
├── coverage_audit/              # small audit CSVs may be committed
└── eda_output/                  # generated EDA charts/reports
```

The source archives are intentionally excluded from Git because of size and upstream terms. Place the files under `raw/` using the indicated names; the original eafc-db-kit test can be found at the upstream project, and the `eafc_db_output.zip` input is our saved export of its six tables for each edition. The previously generated processed output can be shared separately as a compressed GitHub Release asset or team-drive file, rather than committing a >100 MB CSV.

## Reproduce the pipeline (Windows PowerShell)

Run from the repository root, with Python 3.10+:

```powershell
python -m pip install -r requirements.txt

# 1. Create consistent player/club/coach/club-feature tables
python build_dataset.py --root .
python check_outputs.py

# 2. Restore tactics and apply evidence-based age completion
python finalize_dataset.py --input output --out output_final --history-teams raw/history/male_teams.csv --fc24-zip raw/fc24/24.zip --eafc-zip raw/eafc/eafc_db_output.zip

# 3. Audit UCL club coverage
python audit_ucl_coverage.py --input output_final --out coverage_audit --roster-csv reference/historical_ucl_teams_2014_2024.csv --known-map reference/ucl_25_27_verified_mapping.csv --history-teams raw/history/male_teams.csv --fc24-zip raw/fc24/24.zip --eafc-zip raw/eafc/eafc_db_output.zip --offline

# 4. Descriptive EDA (does not require match results)
python run_eda.py --data output_final --coverage coverage_audit/ucl_teams_coverage.csv --out eda_output
```

Primary outputs under `output_final/`: `players_2015_2027.csv`, `clubs_2015_2027.csv`, `coaches_2015_2027.csv`, and `club_features_2015_2027.csv`. Some season-specific fields are legitimately missing; source/snapshot limitations are documented in the audit files. Financial data remains in EUR, not USD.

## Next steps

1. Join season-specific UCL fixtures/results (being collected separately by the team) to the verified club-ID mapping.
2. Construct one match-level sample per game, with home/away club ratings, **position-constrained** best XI player attributes, remaining-squad aggregates, and available contextual features.
3. Train time-split win/draw/loss and score models, then simulate the remaining 2026–27 schedule and knockout bracket to estimate title probabilities.

*Research/coursework use. EA SPORTS, FIFA, SoFIFA, UEFA, and dataset owners retain their respective rights. Please consult source-specific licenses/terms before redistributing raw datasets.*
