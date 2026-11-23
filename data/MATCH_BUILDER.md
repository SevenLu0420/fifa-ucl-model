# UCL match-level builder v2 (2026-11-22)

This package **adds** the match-builder code and match input CSVs to the existing `fifa-ucl-model` repository. Do not overwrite existing `data/build_dataset.py`, `data/finalize_dataset.py`, or original FIFA data. The previous `build_match_dataset.py` is replaced by the v2 version here.

## Fixed and extended

- Read Baptiste's 2015-16 to 2024-25 multi-sheet XLSX **and** separately provided 2014-15 / 2025-26 CSVs.
- Canonical matches include all 2014-15 through 2026-27 records, including the 2026-27 **unplayed** fixtures.
- Restore all 16 previously reviewed regulation-time scores without editing the original XLSX, and use native 90-minute scores in the two new historical CSVs.
- Historical modeling tables **exclude 2026-27** altogether. Future-fixture tables include the 72 completed and 72 scheduled league matches, with null labels for scheduled matches.
- Preserve the mandatory goalkeeper and soft GK/DEF/MID/ATT (1/4/3/3) quotas; substitutions fill their missing positional slots, not trailing slots.
- Add manually verified name equivalents for new supplementary CSV team names. Use `--known-map` for verified 2024-27 EA Club IDs. Unmatched clubs remain missing; never auto-accept fuzzy suggestions.
- No difference (`diff_*`) features. Inputs are home N features, away N features, then match context. Compact = 93 predictors; Full = 797 predictors when the usual final FIFA schema is available.
- No season-level snapshot leakage filtering (as decided for this initial experiment). Do not interpret game rosters as verified real match lineups.

## Installation

Place these files under the repository root (the ZIP's `data/` folders merge into the existing `data/` folder):

```text
fifa-ucl-model/
  data/
    build_match_dataset.py
    test_match_builder.py
    match_builder/
      __init__.py
      mapping.py
      features.py
    reference/
      ucl_2014_15_matches.csv
      ucl_2025_26_matches.csv
      ucl_2026_27_league_schedule.csv
      ucl_extra_time_90min_verified.csv
      ucl_25_27_verified_mapping.csv        <-- from your EXISTING GitHub repository
    raw/matches/
      resultats_ldc_par_saison.xlsx         <-- the already available Baptiste file
    output_final/
      players_2015_2027.csv                <-- already generated locally
      clubs_2015_2027.csv                  <-- already generated locally
    coverage_audit/
      ucl_teams_coverage.csv               <-- already generated locally
    match_audit/
      team_name_mapping_review.csv         <-- optional, if you have it
```

The v2 builder recomputes Top XI and Bench aggregates instead of ingesting the old `club_features_2015_2027.csv` because that table's Top 11 is calculated using a different rule. Accordingly, v2 no longer accepts the obsolete `--club-features` option.

From the repository root in **Windows PowerShell**:

```powershell
python -m pip install pandas numpy openpyxl
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
  --audit-mapping data/match_audit/team_name_mapping_review.csv `
  --out data/match_dataset
```

The `--audit-mapping` argument is optional and silently ignored if the file does not exist. The `--known-map` file is already checked in to your GitHub repo (the verification of 2024-27 club IDs); **retain it**.

## Outputs

- `matches_canonical.csv`: all 1,766 raw match and fixture rows, scores, labels, and mapping status (independent of whether FIFA club data is available).
- `team_name_mapping.csv`: every (season, match-team-name), mapped/unmapped status, evidence type, untrusted suggestions.
- `team_feature_bank.csv` and `team_feature_quality.csv`: calculated XI/Bench features and squad availability.
- `matches_compact.csv`, `matches_full.csv`: **historical 2014-15 to 2025-26 only**, 90-min labeled and two-team-FIFA-matched.
- `fixtures_2026_27_compact.csv`, `fixtures_2026_27_full.csv`: **2026-27 league phase**, 72 played + 72 scheduled, `prediction_ready` tells whether two valid team feature rows exist. Scheduled fixtures must keep null labels.
- `compact_feature_columns.json`, `full_feature_columns.json`: the **only** model-input columns. Do not select all numeric columns from CSVs; many other columns are labels/IDs.
- `sample_status.csv`, `build_report.json`: exclusions and detailed counts.

Expected source-only checks before FIFA filtering: **1,308 + 125 + 189 = 1,622 historical matches**, plus **144 future-season fixtures**, total **1,766 canonical**. There are **21 historical extra-time matches**, of which 16 were corrected from Baptiste's XLSX; 9 of those 16 need goal adjustments, 4 change W/D/L. The actual number of model-ready rows WILL be lower than 1,622 if the real FIFA database lacks clubs or valid XIs. The figure 1,622 was achieved only with artificial, complete FIFA inputs in the unit integration test; it is not a claimed real coverage result.

Data split for subsequent evaluation (recommended): train 2014-15 through 2023-24; validation 2024-25; test 2025-26. Reserve all 2026-27 completed scores for conditioning the league simulation / optional future forward evaluation; do not leak them into the historical test.

## Checks performed here

- Python syntax compilation passed.
- Four unit tests passed: missing-position slot stability, verified Inter mapping without mapping missing clubs, extra-time correction, and future fixture with no target.
- End-to-end test with the **user-provided real match source files and intentionally synthetic, complete FIFA player/club tables** succeeded: 1,766 canonical rows; 1,622 historical features/labels; 144 future fixture feature rows; 21 resolved ET matches; 9 altered historical 120-minute scorelines; 4 corrected WDL outcomes; 72 unplayed rows without labels; 93/797 predictor columns.
- Final real-data FIFA coverage, excluded teams, and exact trainable sample count **cannot be determined here** because the private/generated `players_2015_2027.csv`, `clubs_2015_2027.csv`, and coverage CSV were not uploaded. Run the script against those local files and check `build_report.json`.

`ucl_extra_time_90min_verified.csv` was transcribed from the previously supplied 16-game verification in the conversation. It is reproducible and score-key validated, but this packaging step does not constitute a fresh independent source-level verification of all 16 scores.

## Git

Keep build outputs local if large. Add the generated folder to `.gitignore` if desired:

```gitignore
/data/match_dataset/
```

Commit the small source code and reference CSVs, not the full generated match feature CSVs.
