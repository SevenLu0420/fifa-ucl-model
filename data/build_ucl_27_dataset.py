from pathlib import Path
import numpy as np
import pandas as pd

FIXTURES = Path('match_dataset/fixtures_2026_27_full.csv')
PROXIES = Path('output_final/proxy_profiles.csv')
OUTPUT = Path('match_dataset/fixtures_2026_27_top5.csv')
TEAM_OUTPUT = Path('match_dataset/team_profiles_2026_27_top5.csv')
TARGETS = ('Sabah', 'Slovan Bratislava')
PROXY_METADATA = {'team', 'k', 'n_estimated_donors', 'n_features', 'mean_elo_gap', 'mean_golo_gap'}


def integrate_top5(fixtures: pd.DataFrame, proxy_profiles: pd.DataFrame):
    matches = fixtures.copy()
    proxies = proxy_profiles.loc[proxy_profiles['k'].eq(5)].set_index('team')
    if set(TARGETS) != set(proxies.index):
        raise ValueError(f'Top-5 profiles must contain exactly {TARGETS}; got {list(proxies.index)}')

    side_cols = {side: {c[len(side)+1:]: c for c in matches if c.startswith(side + '_')}
                 for side in ('home', 'away')}
    shared_attrs = set(side_cols['home']) & set(side_cols['away'])
    # Most model attributes are shared; club_id and team identity cannot be averaged.
    attrs = sorted(shared_attrs - {'team', 'club_id'})
    missing = sorted(set(attrs) - set(proxies.columns))
    if missing:
        raise ValueError(f'Proxy profiles lack {len(missing)} actual fixture feature fields: {missing[:15]}')

    score_cols = ['status', 'label_90min_home_goals', 'label_90min_away_goals',
                  'label_90min_goal_diff', 'label_90min_wdl',
                  'label_90min_home_win', 'label_90min_over25']
    before = matches[score_cols].copy(deep=True)
    matches['proxy_feature_source'] = 'original_fc27'
    matches['uses_top5_proxy'] = False
    for team in TARGETS:
        profile = proxies.loc[team]
        bad = [a for a in attrs if pd.isna(profile[a])]
        if bad:
            raise ValueError(f'{team} contains {len(bad)} missing fixture features: {bad[:15]}')
        for side in ('home', 'away'):
            mask = matches[f'{side}_team'].eq(team)
            if mask.any():
                columns = [side_cols[side][a] for a in attrs]
                matches.loc[mask, columns] = np.tile(profile[attrs].to_numpy(dtype=float), (int(mask.sum()), 1))
                matches.loc[mask, 'uses_top5_proxy'] = True
                matches.loc[mask, 'proxy_feature_source'] = 'elo_golo_top5_estimated_positions'

    matches['original_prediction_ready'] = fixtures['prediction_ready'].copy()
    # Prediction readiness reflects completed numeric side-feature coverage;
    # mapping provenance flags remain unchanged for auditability.
    crucial = ['club_overall', 'club_attack', 'club_midfield', 'club_defence',
               'att1_attacking_finishing', 'def1_mentality_interceptions']
    for side in ('home', 'away'):
        miss = [c for c in crucial if c not in side_cols[side]]
        if miss:
            raise ValueError(f'Missing existing {side} columns: {miss}')
    complete = np.ones(len(matches), dtype=bool)
    for side in ('home', 'away'):
        complete &= matches[[side_cols[side][a] for a in crucial]].notna().all(axis=1).to_numpy()
    matches['prediction_ready'] = complete

    pd.testing.assert_frame_equal(before, matches[score_cols], check_dtype=True)
    assert len(matches) == len(fixtures)
    assert matches.match_id.equals(fixtures.match_id)
    assert not matches.match_id.duplicated().any()
    assert matches['prediction_ready'].all(), 'Some fixture profiles remain incomplete'

    team_rows = []
    for team in sorted(set(matches.home_team) | set(matches.away_team)):
        options = []
        for side in ('home', 'away'):
            r = matches.loc[matches[f'{side}_team'].eq(team)]
            if len(r):
                options.append({attr: r.iloc[0][side_cols[side][attr]] for attr in attrs})
        row = {'team': team, **options[0],
               'feature_source': 'elo_golo_top5_estimated_positions' if team in TARGETS else 'original_fc27'}
        team_rows.append(row)
    return matches, pd.DataFrame(team_rows)


def main():
    matches, team_bank = integrate_top5(pd.read_csv(FIXTURES), pd.read_csv(PROXIES))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(OUTPUT, index=False)
    team_bank.to_csv(TEAM_OUTPUT, index=False)
    print(f'Wrote {OUTPUT}: {len(matches)} fixtures, {matches.uses_top5_proxy.sum()} proxy-related matches')
    print(f'Wrote {TEAM_OUTPUT}: {len(team_bank)} team profiles')
    print('Current snapshot results: ', matches.status.value_counts().to_dict())
    print('Prediction-ready: ', matches.prediction_ready.sum(), '/', len(matches))
    print('IMPORTANT: results are NOT updated; refresh past-scheduled games before current as-of forecasting.')


if __name__ == '__main__':
    main()
