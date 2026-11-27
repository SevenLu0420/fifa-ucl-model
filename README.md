# Predicting UEFA Champions League Outcomes with EA Sports FC Data

**Columbia University · IEOR E4523 Data Analytics & Machine Learning · Group 17**  
Jiaxuan Liu · Baptiste Pinatel · Wilson Zhen · Xiaonuo Shi

## 1. Research question and final deliverables

**Can EA Sports FC (FIFA) club and player ratings help predict real UEFA Champions League match outcomes, and what do these predictions imply for the 2026-27 competition?**

We combine season-specific game ratings with historical UCL fixtures, construct match-level features, compare multiple prediction targets and feature sets under **chronological evaluation**, interpret selected models, and simulate the 2026-27 competition. The final outputs are:

- **Three W/D/L probability forecasters:** direct classification, goal-difference regression, and paired goal-count regression; plus their equal-weight **Soft Vote**.
- **Held-out 2025-26 evaluation** against Always Home Win and Overall Rating Difference baselines, with rolling-year evidence for model/feature choices.
- **2026-27 advancement and title probabilities** from three separate 20,000-run Monte Carlo simulations.
- **One representative knockout bracket** conditioned on the most likely Soft Vote champion, with match-up advancement probabilities and conditional reference scores.

## 2. Data and feature construction

### Sources and coverage

We consolidated **13 EA game editions, FIFA 15-FC 27**, covering the corresponding **2014-15 through 2026-27** team/player snapshots: **234,068 player records** and **8,557 club-season records**. Historical game-edition data were sourced from SoFIFA-derived collections, FC 25-27 game databases, and supplementary EA FC ratings records; see the source list below.

| Period / role | Source | Use |
|---|---|---|
| FIFA 15-23 | [SoFIFA historical Kaggle data](https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset) | Historical player/club ratings |
| FC 24 | [EA Sports FC 24 Kaggle data](https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset) | Roster and club attributes |
| FC 25-27 | [eafc-db-kit](https://github.com/ViniMacacari/eafc-db-kit) | Native players, clubs, managers and squad associations |
| FC 25 supplemental | [FC25 Ratings and Stats](https://www.kaggle.com/datasets/nyagami/ea-sports-fc-25-database-ratings-and-stats); [SoFIFA June 2025](https://www.kaggle.com/datasets/aniss7/fifa-player-data-from-sofifa-2025-06-03); [FC25 SoFIFA Merge](https://www.kaggle.com/datasets/sametozturkk/ea-sports-fc-25-real-player-data-sofifa-merge) | Identity and financial data enrichment |
| FC 26 supplemental | [FC26 Player Data](https://www.kaggle.com/datasets/rovnez/fc-26-fifa-26-player-data) | Position-specific and financial attribute cross-checks |
| FC 26-27 ratings | [EA SPORTS FC Ratings](https://www.ea.com/games/ea-sports-fc/ratings) | Available supplementary player ratings / identities |
| UCL teams and results | [UEFA Champions League](https://www.uefa.com/uefachampionsleague/) and [OpenFootball](https://github.com/openfootball/champions-league) | Schedule/results reconciliation and club mapping |

The canonical match table contains **1,766 rows**: **1,622 historical results** for 2014-15 through 2025-26 and **144** fixtures for 2026-27. Of the historical rows, **1,433 matches** have usable matched squad attributes and form the modeling sample.  
The original historical UCL club mapping covers **397 / 428 team-season entries (92.8%)** in available game data.

### Targets and features

- **Targets:** 90-minute home/draw/away (`H/D/A`), goal difference, home goals, away goals; auxiliary home-win and over-2.5 binary outcomes. Extra time and penalties are **not** folded into the 90-minute label.
- **Estimated strongest XI:** 1 GK + 4 DEF + 3 MID + 3 ATT selected from a *season-level* roster (not observed match-day starters). Position aggregates, reserve depth and club ratings are extracted consistently for home/away teams.
- **Feature groups:** Club Only (15); Club + DEF position core (31); Club + position-aware core (61); Club + compact squad information (91), with larger individual-player feature sets retained for ablations. Matches use `[home features | away features | pre-match venue/stage context]`.
- **Preprocessing:** season-aware splits; train-only imputation and scaling; structural missingness audit; exclusion of IDs, post-match data and historically unavailable player fields from inappropriate training use.

**Edition-level harmonization and coverage audits** (optional if `data/output_final/` is already prepared; start in the repository root; paths below are relative to `data/`):

```powershell
cd data
python build_dataset.py --root .
python check_outputs.py
python finalize_dataset.py --input output --out output_final --history-teams raw/history/male_teams.csv --fc24-zip raw/fc24/24.zip --eafc-zip raw/eafc/eafc_db_output.zip
python audit_ucl_coverage.py --input output_final --out coverage_audit --roster-csv reference/historical_ucl_teams_2014_2024.csv --known-map reference/ucl_25_27_verified_mapping.csv --history-teams raw/history/male_teams.csv --fc24-zip raw/fc24/24.zip --eafc-zip raw/eafc/eafc_db_output.zip --offline
python run_eda.py --data output_final --coverage coverage_audit/ucl_teams_coverage.csv --out eda_output
cd ..
```

**Match-level builder** (from the repository root; requires the standardized Club and Player tables):

```powershell
python -m pip install -r requirements.txt
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

The principal outputs are `data/match_dataset/matches_full.csv`, `fixtures_2026_27_full.csv`, and `team_feature_bank.csv`. Historical preprocessing and construction scripts also produce coverage audits and structured clean tables under `data/output_final/`.

## 3. Modeling experiments and final selection

### Evaluation protocol

A robust model selection compares expanding-train / held-out **2021-22, 2022-23, 2023-24 and 2024-25** folds. Selected models are refit through 2024-25 and evaluated on the untouched **2025-26 test season (174 matched matches)**.

**What was tried, and why it matters:**

- Six targets and a wide algorithm search (linear / logistic, Poisson, KNN/SVM, MLP, random forests, boosting). Regression routes are converted to **the same three-way A/D/H probabilities** for evaluation; binary tasks remain auxiliary.
- Club Only versus Compact / Full / player-only / player+club / position-aware inputs, including grouped permutation importance and partial correlations by GK/DEF/MID/ATT. Very large player sets overfit; player information is interpretable but does **not** uniformly improve out-of-time performance.
- Original ratings versus **difference-only / difference+level** features and club-residualized player features. Gains were small or inconsistent; residualization was not adopted as a universal enhancement.
- Draw-diagnostic confusion matrices, probability calibration and validation-chosen **hard-label draw thresholds**; thresholds affect the displayed class decision, **not the probabilities used for Monte Carlo**.
- Chronological **Hard Vote**, equal-probability **Soft Vote**, probability Stacking and Stacking + pre-match context. The simple Soft Vote was retained for interpretability and competitive validation metrics; more complex stacking did not consistently win.
- Global and local **SHAP** for the selected routes, plus concrete **case studies**. Defensive quality and midfield abilities repeatedly matter within fitted models.

### Selected models

| Prediction route | Final algorithm | Feature set | Features | Probability generation |
|---|---|---|---:|---|
| **Direct W/D/L** | Logistic Regression | Club + DEF | 31 | Three-class `predict_proba`, ordered A/D/H |
| **Goal Difference** | Ridge Regression | Club + DEF | 31 | Predicted goal difference + training-residual Normal CDF, cut at ±0.5 |
| **Two Goals** | Random Forest Regressor (home and away goals) | Club + position core | 61 | Independent Poisson home/away goal distributions; Skellam A/D/H probabilities |
| **Soft Vote** | Equal-weight probability average of the three routes | — | — | Mean A/D/H probabilities |

The score-generating simulation uses the **Two Goals** Poisson score matrix. For Goal Difference and Soft Vote, the H/D/A regions are reweighted to the route's target outcome probabilities while **preserving relative score probabilities within each outcome region**. This supplies scorelines for points, goal difference, and aggregate knockout ties without confusing regression output with a direct score prediction.

**Final held-out 2025-26 test metrics**:

| Method | Accuracy | Macro-AUC | Macro-F1 | Brier | Log loss |
|---|---:|---:|---:|---:|---:|
| Always Home Win | 0.4943 | 0.5000 | 0.2205 | 1.0115 | 10.4807 |
| Overall Diff Logistic | **0.5805** | 0.6581 | 0.4116 | 0.5634 | 0.9665 |
| Two Goals | 0.5747 | 0.6729 | 0.4101 | **0.5582** | **0.9516** |
| Soft Vote | 0.5632 | **0.6875** | 0.3997 | 0.5584 | 0.9552 |
| Goal Difference | 0.5575 | 0.6659 | **0.4440** | 0.5647 | 0.9631 |

**Why keep three routes:** Two Goals has strong proper probability scores and an interpretable scoreline generator; Soft Vote provides the strongest Macro-AUC; Goal Difference has the best Macro-F1 among these three. They represent complementary uncertainties for the competition forecast.  

## 4. FC27 missing-club profiles: Sabah and Slovan Bratislava

These two 2026-27 participants lack usable FIFA club/player attributes. We select similar available FC27 **donor clubs using ClubElo `Elo` (general strength) and `Golo` (goal-scoring strength)**; use a distance-weighted average of their team-level features to create proxy profiles. **Top 5** was selected over **Top 24** after sensitivity analysis. The compared advancement and championship probabilities were generally similar.

Generate the proxy comparison from the repository root:

```powershell
python ./data/compare_proxy_top5_top24.py --elo clubelo_candidates.csv --estimate-missing --out model/results/proxy_top5_top24
```

This generates `model/results/proxy_top5_top24/proxy_profiles.csv`. Ensure the `PROXIES` variable in `build_ucl_27_dataset.py` points to that file, then:

```powershell
python ./data/build_ucl_27_dataset.py
```

The **fixed final simulation inputs** are `data/match_dataset/fixtures_2026_27_top5.csv` (144 fixtures) and `team_profiles_2026_27_top5.csv` (36 teams).

## 5. Monte Carlo tournament simulation

### Rules and modeling assumptions

`ucl_simulator.py` builds a prediction cache for all plausible opponent pairs **once**, then runs each model method separately. In each simulated tournament:

1. **League phase:** keep recorded 90-minute scores; draw scores for remaining games from the precomputed score distributions. The fixture list is fixed: 36 clubs, 8 matches per club, 144 total league matches.
2. **League ranking:** award 3/1/0 points and apply implemented UEFA tie-break statistics. Places **1-8** qualify directly for the Round of 16, **9-24** enter knock-out play-offs, and **25-36** are eliminated. For remaining unresolved ties, the implementation falls back to **alphabetical order**; it does not have complete discipline/coefficient information.
3. **Play-offs / Round of 16:** sample UEFA-style **ranking-constrained** match-ups within the existing fixed bracket framework; simulate home and away 90-minute legs and compare aggregate goals. No away-goals advantage is applied.
4. **Quarter-finals / semi-finals:** continue along the preassigned bracket; sample both legs. On aggregate ties, approximate extra-time scoring using **1/3 of 90-minute Poisson intensity**, then a **50/50** penalty shoot-out if still tied.
5. **Final:** single match at a neutral venue; resolve a tied 90-minute result by extra time/penalties. Record the champion and all round advancements.

### Championship probabilities and final bracket

- Run **10,000~20,000 simulated tournaments per method** (`two_goals`, `soft_vote`, `goal_difference`). The predicted title probability is the fraction of simulations won, and qualification odds are analogous frequencies.
- Let the champion with the highest simulated Soft Vote title probability be the **predicted champion**. Among paths in which that team wins, select a *representative knockout path* using a heuristic score: **50% knockout-round composition representativeness + 30% individual tie-advancement confidence + 20% path likelihood** (just a customized reward function).
- Plot one fully connected play-off-to-final bracket. The percentage next to a highlighted team is its **unconditional model advancement probability for that particular tie**; the reference score is the most likely regulation-time score **conditional on that advancement outcome** (aggregate for two legs).

In the provided 20,000-run snapshot, the **Soft Vote** projected **Paris Saint-Germain** as the most possible champion (**25.04%**, 5,007 title-winning simulations); **Two Goals** gave PSG **20.12%**, and **Goal Difference** **26.82%**. These results are conditional on the archived (partly stale) match snapshot.

## 6. How to run

Run notebooks in VS Code / Jupyter from the **repository root** using the project Python environment. The final notebooks are designed to execute from a restarted kernel, in cell order:

1. `UCL_Exploration.ipynb`: missingness, exploratory statistics, initial model search, draw/feature/position ablations. Generates `analysis/eda_output/prepared/` and initial modeling artifacts.
2. `UCL_Prediction_and_Analysis.ipynb`: rolling model selection, 2025-26 held-out evaluation, SHAP, ensemble comparisons, Top 5 proxy sensitivity, re-fitting and **the complete 2026-27 simulation**.

The final simulation controls in `UCL_Prediction_and_Analysis.ipynb` are:

```python
N_SIM = 20_000        # start with 100 for a smoke test
SEED = 42
BRACKET_METHOD = 'soft_vote'

# After loading fixtures/models and building cache once:
result = sim.run_monte_carlo(
    fixtures, cache, n=N_SIM, method='soft_vote', seed=SEED,
    tie_break='name', representative=True,
    bracket_weights=(0.5, 0.3, 0.2)
)
fig, bracket_table = sim.plot_representative_knockout_bracket(
    selection=result['representative'], cache=cache,
    championship_odds=result['odds']
)
```

### Final generated artifacts

Under `model/results/ensemble_final/`:

- `model_wdl.joblib`, `model_goal_difference.joblib`, `model_two_goals.joblib` — final fitted model bundles with feature-column metadata.
- Final held-out metrics and comparison outputs are generated in model result folders.

Under `model/results/ucl_2026_27_top5/`:

- `odds_two_goals.csv`, `odds_soft_vote.csv`, `odds_goal_difference.csv`: Top 8 / Top 24 / each knockout round / title probabilities.
- `league_*.csv`: league-phase ranking summaries (context only; not the main presentation target).
- `champion_model_comparison.csv`: three-model title-probability comparison.
- `representative_bracket_matches.csv`, `representative_bracket.png`, `representative_bracket.svg`: one presentable knockout tree and its tabular reference.

## 7. Limitations and reproducibility

- FIFA ratings are proxies for ability, and precomputed attributes may not capture recent injuries, transfers, form, tactical match-ups, or refereeing.  
- The missing-club player statistics are estimated, not observed.  
- Two Goals assumes conditionally independent Poisson scoring; other routes retain that within-class score structure.  
- Neutral-venue symmetrization, extra-time scoring and shoot-outs are modeling assumptions.

---

*Coursework use. FIFA, EA SPORTS FC, SoFIFA, UEFA and data owners retain their rights. Follow the original sources' licensing conditions for redistribution.*
