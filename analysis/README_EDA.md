# UCL Match EDA — reproducible results (2026-11-23)

## Install and run

Copy `analysis/run_match_eda.py`, `analysis/missing_preprocessing.py` to the repository's `analysis/`; replace `data/build_match_dataset.py` and `data/match_builder/mapping.py` with the included fixes. **Do not modify FIFA source or processed CSVs.**

From repository root:

```powershell
python -m pip install pandas numpy matplotlib scikit-learn
python data/test_match_builder.py
.\run_match_builder.bat
python analysis/run_match_eda.py --data data/match_dataset --out analysis/eda_output
```

Outputs: `analysis/eda_output/figures/` has nine paired PNG/PDF figures; `tables/` has CSV statistics and missingness audits; `summary.json` contains numeric results. `missing_preprocessing.py` provides a training-only imputation pipeline; EDA does not impute anything.

## Builder corrections

* Explicitly maps `SSC Napoli` to the verified EA club ID **48**, conditional on presence in that season's Club table. Expect up to 8 additional 2025–26 matches depending on the opponent's availability.
* Corrects neutral-ground flags for the 10 finals incorrectly marked non-neutral; additionally accounts for the neutral 2019–20 quarter-finals and semi-finals. Review all abnormal neutral-site matches separately if needed.
* Sabah and Slovan Bratislava are absent from the uploaded `clubs_2015_2027.csv` in *all* versions, including FC27. They are real 2026–27 UCL teams according to UEFA, but should not be assigned arbitrary FIFA IDs. Simulation needs separate, explicitly labeled surrogate strength estimates with sensitivity analysis; the current builder will continue to mark the 16 related games `prediction_ready=False`.

## Historical sample (before Napoli fix)

* 1,425 matches, 2014–15 through 2025–26; home wins **672 (47.16%)**, draws **286 (20.07%)**, away wins **467 (32.77%)**.
* 90-min goals: home **1.714**, away **1.344**, total **3.058** per game. Goal difference mean: **+0.370**.
* Even-strength subset with `|home_club_overall - away_club_overall| <= 2`: **446** matches; **45.07%** home wins, **23.32%** draws, **31.61%** away wins. This is consistent with an association with home status, not a causal estimate of venue effect. Neutral-ground misclassification in the original dataset is a further reason to refrain from causal language.
* Relative club Overall vs 90-min goal difference: Pearson **r=0.496**; XI mean-rating difference **r=0.490**; bench mean-rating difference **r=0.403**. The associations overlap strongly because these ratings correlate with one another.
* Rating gap <= -10: away wins **68.2%** (n=85); gap +5 to +10: home wins **76.8%** (n=190). This motivates comparing relative-strength measures without feeding any post-match feature into training.
* League matches: 1,055; 47.39% home wins; mean goals 3.080. Knockout matches: 370; 46.49% home wins; mean goals 2.995. Differences are descriptive and influenced by team selection and competition structure.
* Aggregate total-goals mean **3.058**, sample variance **3.306**. A Poisson goals model is a reasonable benchmark, but marginal overdispersion suggests evaluating calibration and potential alternatives.
* A same-sample illustrative OLS fit (NOT out-of-sample evidence) explaining 90-min goal difference has R^2 **0.24621** with club Overall difference, **0.24632** with added XI mean rating difference, and **0.24646** with added Bench mean rating difference. These features are redundant as linear proxies; **do not conclude individual player attributes lack predictive signal**. Full 797-dimensional vs Compact 93-dimensional comparisons require time-based held-out seasons and appropriate regularization.

## Missing values and preprocessing

* Compact: only `home_bench_att_best_overall` (222/1425 = 15.58%) and `away_bench_att_best_overall` (224/1425 = 15.72%) are missing. These generally represent *no remaining eligible attacker after selecting XI*, not randomly missing ratings. Recommended: add explicit absence indicator and training-median fill only for algorithms requiring a complete numeric matrix. For interpretability, a separate sensitivity experiment can encode missing bench attacker strength as 0 **with an indicator**, since no attacker differs from a weak attacker.
* Full: adds **22** `*_mentality_composure` positions, each 188/1425 = 13.19% missing, entirely from FIFA15–16. This is **version-level missingness**. Better core-model design: drop those 22 columns in the primary Full analysis and run a sensitivity experiment retaining them with train-only imputation and version/missing flags. Median-filling 2014–16 values does not reconstruct the true values and may induce version shortcuts.
* Other predictors: no NaNs in the current sample. 20 numeric predictors are constant (confirm counts from exported missingness tables). Training preprocessing should remove constants and train-only fit MedianImputer + MissingIndicator/VarianceThreshold. Pipeline in `analysis/missing_preprocessing.py` illustrates this, but for the primary Full model remove the 22 `mentality_composure` slots first.
* All preprocessing must be fitted on train seasons 2014–15 to 2023–24 only; validation 2024–25; test 2025–26. Always use `*_feature_columns.json` to select model inputs, not the entire CSV: some columns contain targets, final scores, and other match metadata.

## Figures and report logic

1. `01_coverage` — show effective sample size, exclusions and sampling bias.
2. `02_outcomes`, `03_goals_by_season` — quantify home/away asymmetry and score frequency.
3. `04_stage_outcomes` — contrast league vs knockout context; differences are not adjusted for team strength.
4. `05_rating_gap_results` — show the strong monotonic relation between relative strength and outcome; every marker includes sample size.
5. `06_strength_correlation` — compare simple proxies: club, starting XI and bench.
6. `07_total_goals` — motivate goals regression, Poisson baseline, and scoring simulation.
7. `08_missingness` — justify feature treatment before models.
8. `09_strength_by_stage` — inspect the same relationship in two stage strata; endpoint bins have tiny n in knockouts.

## Important caveats

These descriptive statistics are calculated on **eligible matches only**, prior to the Napoli/neutral-venue fixes. Re-run the script after rebuilding the datasets; the correct published numbers are the *updated* CSVs/plots. Distinguish associations from causal claims. Never use `label_90min_*`, `extra_time`, `has_penalties`, actual attendance, or final results as model predictors. Also, treating season-level FIFA roster snapshots as pre-game information is an approximation, not match-day starting XI.
