# Predicting UEFA Champions League Outcomes with EA Sports FC Data

**Columbia University · IEOR E4523 Data Analytics & Machine Learning · Group 17**  
Jiaxuan Liu · Baptiste Pinatel · Wilson Zhen · Xiaonuo Shi

## 1 Research objective and workflow

**Research question:** *Can EA Sports FC (formerly FIFA) player statistics predict real UEFA Champions League (UCL) match outcomes?*

We combine historical FIFA/EA FC player and club ratings with actual UCL fixtures and results to evaluate whether detailed player attributes and squad composition add predictive value beyond the game's overall club ratings. The planned final application is a probabilistic simulation of the **2026-27 men's Champions League**, using completed results where available.

**Pipeline:** Game data collection → cross-edition standardization → UCL club matching → match-level feature construction → predictive models and interpretation → tournament simulation.

## 2 Data collection and preprocessing

*Completed by Jiaxuan Liu*  

We integrated **13 game editions (FIFA 15-FC 27)**, corresponding to the **2014–15 through 2026–27 seasons**. The resulting tables contain **234,068 player observations** and **8,557 club-season observations**, with coach associations and squad-level aggregates. Detailed technical skills, overall/potential ratings, club attack/midfield/defence ratings, available finances, and selected tactical information are harmonized across versions; unavailable or non-equivalent fields remain missing rather than fabricated. We retain player names for reference but omit cosmetic game fields.

Direct SoFIFA requests and Selenium/Playwright scraping encountered HTTP 403/Cloudflare restrictions; a probe of the [SoFIFA partner API](https://sofifa.com/document) also returned 403 (partner authorization/static-IP allowlisting is required). EA's ratings JSON API provided useful supplementary player attributes but not complete historical squad/club data. We therefore combined publicly available datasets with locally parsed game databases.  

We currently focus on **men's UEFA Champions League (UCL)** to ensure consistent historical coverage and comparability across seasons. Women's club teams were first introduced in EA SPORTS FIFA 23 (2022), with initially limited league and club coverage. As a result, women's player and club data are considerably sparser and less consistent across the 2014-2027 study period. We therefore exclude the women's competition from the current analysis while leaving it as a potential extension for future research.

### Data sources

### Data Sources

| Edition / Role | Source | Usage |
|---|---|---|
| FIFA 15–23 | [Kaggle: SoFIFA Historical Dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset) | Historical player attributes, club ratings, coaches, contracts, and roster updates |
| FC 24 | [Kaggle: EA Sports FC 24 Dataset](https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset) | Player, club, and coach data |
| FC 25–27 | [GitHub: eafc-db-kit](https://github.com/ViniMacacari/eafc-db-kit) | Primary game database: players, clubs, managers, leagues, and squad relationships |
| FC 25 | [Kaggle: FC25 Ratings and Stats](https://www.kaggle.com/datasets/nyagami/ea-sports-fc-25-database-ratings-and-stats) | Supplementary player identities and six summary ratings |
| FC 25 | [Kaggle: SoFIFA June 2025](https://www.kaggle.com/datasets/aniss7/fifa-player-data-from-sofifa-2025-06-03) | Additional market values, wages, release clauses, and contract information |
| FC 25 | [Kaggle: FC25 SoFIFA Merge](https://www.kaggle.com/datasets/sametozturkk/ea-sports-fc-25-real-player-data-sofifa-merge) | Additional player identities and financial attributes |
| FC 26 | [Kaggle: FC26 Player Data (rovnez)](https://www.kaggle.com/datasets/rovnez/fc-26-fifa-26-player-data) | Additional financial attributes, position-specific ratings, and career information |
| FC 26–27 | [EA SPORTS FC Ratings](https://www.ea.com/games/ea-sports-fc/ratings) | Retrieved ratings API data for supplementary player attributes, identities, and six summary ratings |
| UCL Club Coverage | [OpenFootball UCL](https://github.com/openfootball/champions-league) and [UEFA](https://www.uefa.com/uefachampionsleague/) | Historical participants, club-name reconciliation, and season-specific coverage validation |

**Integration strategy:** Historical SoFIFA-style datasets serve as the primary source for FIFA 15-FC 24, while native EA game databases serve as the primary source for FC 25-27. Supplementary datasets are merged using player IDs or conservative identity matching to maximize attribute coverage without overwriting validated game ratings.

Additional sources were reviewed for coverage and consistency.

### Codes and outputs

The pipeline selects edition-specific snapshots (preferring the last update before UCL kick-off when available), standardizes player/team/coach schemas, performs conservative cross-season age completion, restores available tactical fields, and audits UCL club mapping. Of **428 men's UCL team-season entries**, **397 (92.8%)** match the available game data; **31** are flagged as unmatched.

```text
fifa-ucl-model/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   ├── build_dataset.py
│   ├── check_outputs.py
│   ├── finalize_dataset.py
│   ├── audit_ucl_coverage.py
│   ├── run_eda.py
│   ├── field_dictionary.csv
│   ├── FINAL_COLUMNS.md
│   ├── reference/
│   │   ├── historical_ucl_teams_2014_2024.csv
│   │   └── ucl_25_27_verified_mapping.csv
│   ├── raw/                 # Local inputs
│   ├── output/              # Generated intermediate tables
│   ├── output_final/        # Generated final tables
│   ├── coverage_audit/      # UCL matching reports
│   └── eda_output/          # EDA figures and tables
├── model/                   # Planned: match features and model training
├── analysis/                # Planned: evaluation and ablations
└── simulation/              # Planned: 2026–27 Monte Carlo simulation
```

Place downloaded inputs under `data/raw/`; large datasets are not committed. From **the repository root**:

```powershell
python -m pip install -r requirements.txt
cd data
python build_dataset.py --root .
python check_outputs.py
python finalize_dataset.py --input output --out output_final --history-teams raw/history/male_teams.csv --fc24-zip raw/fc24/24.zip --eafc-zip raw/eafc/eafc_db_output.zip
python audit_ucl_coverage.py --input output_final --out coverage_audit --roster-csv reference/historical_ucl_teams_2014_2024.csv --known-map reference/ucl_25_27_verified_mapping.csv --history-teams raw/history/male_teams.csv --fc24-zip raw/fc24/24.zip --eafc-zip raw/eafc/eafc_db_output.zip --offline
python run_eda.py --data output_final --coverage coverage_audit/ucl_teams_coverage.csv --out eda_output
```

## 3. Match-level dataset construction

The **match-level dataset builder** uses the standardized FIFA 15–FC 27 Player/Club tables.  

**Match sources and coverage.** Baptiste's original workbook provides 1,308 UCL matches (2015–16 to 2024–25); supplementary datasets collected by Jiaxuan provide 125 matches for 2014–15 and 189 for 2025–26. These total **1,622 historical results spanning 12 seasons**. A separate 2026–27 league-phase file contains 144 fixtures, including 72 completed and 72 scheduled matches as of 2026-11-22. The canonical output preserves all **1,766** rows, while historical training sets exclude 2026–27 entirely.

**Match labels.** The principal targets use **90-minute scores (including stoppage time, excluding extra time and penalties)**. All 16 extra-time matches in the original workbook are reconciled using a separate reviewed correction CSV (9 changed scores and 4 changed W/D/L labels). The builder retains original final scores, extra-time/penalty indicators, and match provenance. Targets include home/draw/away classification, goal difference, and separate home/away goal counts, with optional derived binary labels.

**Team features.** For each `(season, club_id)`, the builder selects an estimated strongest XI with soft positional quotas of **1 GK + 4 defenders + 3 midfielders + 3 attackers**, ranking within each group by Overall and filling unavailable positional slots without duplicating players. This estimated XI is a season-level roster representation—not an observed match-day lineup. Other squad members are represented by fixed-width bench/depth aggregates. Club ratings are included. The builder deliberately recomputes its Top XI and bench statistics instead of reusing the upstream `club_features` aggregates, whose selection rule differs.

**Feature layout and datasets.** Both variants follow **[Home Team Features | Away Team Features | Match Context]**, with identical feature ordering for the two sides and **no engineered home-away difference columns**. The compact variant has **93 predictors** (club ratings, XI/bench summaries, context); the full variant has **797 predictors** (compact plus individual XI attributes for both sides). Club identities/names, actual attendances, referees, and post-match information are excluded from the current predictors. Club matching uses verified season-specific mappings, explicit aliases, and conservative Club ID linkage; unresolved teams are reported and excluded from model-ready feature rows rather than assigned fabricated scores.

**Generated files** (`data/match_dataset/`): `matches_canonical.csv`, `team_name_mapping.csv`, `team_feature_bank.csv`, `team_feature_quality.csv`, `matches_compact.csv`, `matches_full.csv`, `fixtures_2026_27_compact.csv`, `fixtures_2026_27_full.csv`, `compact_feature_columns.json`, `full_feature_columns.json`, `sample_status.csv`, and `build_report.json`. The JSON feature lists define model inputs and prevent accidental inclusion of IDs or targets. Missing values are not globally imputed in this builder; train-only preprocessing will be part of the modeling stage.

**Build from the repository root:**

```powershell
python data/test_match_builder.py
python data/build_match_dataset.py `
  --matches data/raw/matches/resultats_ldc_par_saison.xlsx `
  --historical-csv data/reference/ucl_2014_15_matches.csv `
  --historical-csv data/reference/ucl_2025_26_matches.csv `
  --future-schedule data/reference/ucl_2026_27_league_schedule.csv `
  --corrections data/reference/ucl_extra_time_90min_verified.csv `
  --players data/output_final/players_2015_2027.csv `
  --clubs data/output_final/clubs_2015_2027.csv `
  --coverage data/coverage_audit/ucl_teams_coverage.csv `
  --known-map data/reference/ucl_25_27_verified_mapping.csv `
  --out data/match_dataset
```

**Next: predictive modeling** Train chronological baselines with 2014–15 to 2023–24 for fitting, 2024–25 for validation, and 2025–26 for a held-out test. Compare Club-only, Compact and Full player features under three strategies: direct W/D/L classification, goal-difference regression, and paired goal-count models. Evaluate accuracy, macro-F1, multiclass log loss, probability calibration and task-specific regression metrics; conduct feature-importance and ablation analyses. Fit all imputers/encoders only on training data. The first version uses season-level game-edition rosters without match-date snapshot filtering, a known limitation.

## 4. Planned 2026–27 UCL simulation

Use known match results and fixtures to simulate the remaining league-phase matches, determine the standings and qualification slots under UEFA rules, draw permitted knockout pairings, and simulate each two-legged tie and the final. Repeated Monte Carlo runs will estimate each club's probabilities of reaching successive rounds and winning the title. This stage is **planned, not yet implemented**.

---
*Coursework use. FIFA, EA SPORTS FC, SoFIFA, UEFA and the respective dataset owners retain their rights. Consult original sources and licenses before redistributing data.*
