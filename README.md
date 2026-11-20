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

## 3. Planned match-level dataset and modeling

We will build **one observation per match**, joining both teams to the corresponding season's game ratings. Inputs will include home/away status, club ratings, a **position-constrained best XI** with detailed player attributes, remaining-squad aggregates and depth, available coach information, and optional pre-match context (e.g., form, Elo, referees). We will keep missing-club cases explicit rather than invent FIFA ratings.

Candidate labels include **three-class home win/draw/away win**, **goal difference** (regression), and **home/away goals** (count or score-distribution models). We will compare simple club-overall baselines with engineered squad-feature models, then evaluate richer player-level models if sample size permits. Validation will be split chronologically by season, with accuracy and probability calibration metrics. Feature importance and ablation tests will assess the marginal value of player-level detail; referee variables are exploratory.

## 4. Planned 2026–27 UCL simulation

Use known match results and fixtures to simulate the remaining league-phase matches, determine the standings and qualification slots under UEFA rules, draw permitted knockout pairings, and simulate each two-legged tie and the final. Repeated Monte Carlo runs will estimate each club's probabilities of reaching successive rounds and winning the title. This stage is **planned, not yet implemented**.

---
*Coursework use. FIFA, EA SPORTS FC, SoFIFA, UEFA and the respective dataset owners retain their rights. Consult original sources and licenses before redistributing data.*
