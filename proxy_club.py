from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


TARGETS = {
    'Sabah': {'elo': 1561.0, 'golo': 1.49},
    'Slovan Bratislava': {'elo': 1540.0, 'golo': 1.51},
}

RATINGS = ['overall', 'attack', 'midfield', 'defence',
           'domestic_prestige', 'international_prestige']
CLUB_RATINGS = ['club_' + c for c in RATINGS]

ALIASES = {
    'Cardiff': 'Cardiff City', 'Sirius': 'IK Sirius',
    'Verona': 'Hellas Verona', 'Valladolid': 'Real Valladolid',
    'Molde': 'Molde FK', 'Red Star': 'Red Star FC (FRANCE)',
    'Shanghai Port FC': 'Shanghai Port', 'Metz': 'FC Metz',
    'Mechelen': 'KV Mechelen', 'Erzurum': 'Erzurumspor FK',
    'Tromso': 'Tromsø IL', 'Karlsruhe': 'Karlsruher SC',
    'Annecy': 'FC Annecy', 'Essen': 'Rot-Weiß Essen',
    'Charlton': 'Charlton Athletic', 'Duisburg': 'MSV Duisburg',
    'Thun': 'FC Thun', 'Nancy': 'AS Nancy',
    'Lubin': 'Zagłębie Lubin', 'Basel': 'FC Basel',
    'Lillestrom': 'Lillestrøm SK',
}


def _normalize(name):
    name = unicodedata.normalize('NFKD', str(name))
    name = ''.join(c for c in name if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9]', '', name.casefold())


def load_donors(elo_csv, clubs_csv, season='2026-27', allow_older=False):
    """Match screenshot Elo/Golo candidates to real FIFA club ratings."""
    elo = pd.read_csv(elo_csv)
    clubs = pd.read_csv(clubs_csv)
    needed = {'team', 'elo', 'golo'}
    if not needed.issubset(elo):
        raise ValueError(f'Elo input requires {needed}')
    if not {'season', 'club_id', 'club_name', *RATINGS}.issubset(clubs):
        raise ValueError('Clubs CSV lacks essential club rating columns')

    if 'gender' in clubs:
        clubs = clubs.loc[clubs['gender'].eq(0)]
    clubs = clubs.copy()
    clubs['_name_key'] = clubs.club_name.map(_normalize)
    clubs['_year'] = pd.to_numeric(clubs.season.astype(str).str[:4], errors='coerce')
    clubs = clubs.sort_values(['_year', 'snapshot_date'], ascending=False)
    if not allow_older:
        clubs = clubs.loc[clubs.season.eq(season)]

    rows = []
    for rec in elo.itertuples(index=False):
        if _normalize(rec.team) in {'sabah', 'sabahfk', 'slovan', 'slovanbratislava'}:
            continue
        name = ALIASES.get(rec.team, rec.team)
        found = clubs.loc[clubs['_name_key'].eq(_normalize(name))]
        if found.empty:
            continue
        club = found.iloc[0]
        row = {'elo_name': rec.team, 'fifa_name': club.club_name,
               'elo': float(rec.elo), 'golo': float(rec.golo),
               'season': club.season, 'club_id': club.club_id}
        row.update({'club_' + c: pd.to_numeric(club[c], errors='coerce') for c in RATINGS})
        rows.append(row)

    if not rows:
        raise ValueError('No Elo candidates matched FIFA clubs')
    return pd.DataFrame(rows).drop_duplicates('elo_name').reset_index(drop=True)


def choose_neighbors(donors, target, k, elo_scale=20.0, golo_scale=0.30):
    """Rank actual FIFA donors by combined Elo and Golo distance."""
    if target not in TARGETS:
        raise KeyError(f'Unknown target: {target}')
    ref = TARGETS[target]
    near = donors.copy()
    near['elo_gap'] = near.elo - ref['elo']
    near['golo_gap'] = near.golo - ref['golo']
    near['distance'] = np.hypot(near.elo_gap / elo_scale,
                                near.golo_gap / golo_scale)
    near = near.sort_values(['distance', 'elo_name']).head(k).copy()
    if len(near) != k:
        raise ValueError(f'{target}: requested {k}, found {len(near)} eligible donors')
    near['weight'] = 1 / (near.distance + 0.1)
    near['weight'] /= near.weight.sum()
    near['target'] = target
    near['k'] = k
    return near


def attach_final_features(donors, bank_csv, estimate_missing=False, neighbor_k=5):
    """Attach precomputed team-level attributes, never building individual player slots.

    estimate_missing=False rejects missing donors instead of silently inventing
    attributes. True uses club-rating nearest COMPLETE profiles and records which
    donors were estimated. Use this only when all-club feature bank is unavailable.
    """
    bank = pd.read_csv(bank_csv, low_memory=False)
    if not {'season', 'club_id', *CLUB_RATINGS[:4]}.issubset(bank):
        raise ValueError('Feature bank needs season, club_id, and four club ratings')
    bank = bank.drop_duplicates(['season', 'club_id']).copy()
    excluded = {'club_id', 'fifa_version', 'season', 'club_name', *CLUB_RATINGS}
    feature_cols = [c for c in bank.select_dtypes(include=np.number).columns
                    if c not in excluded and not c.startswith(('label_', 'match_'))]
    if not feature_cols:
        raise ValueError('Feature bank has no precomputed player or position attributes')

    combined = donors.merge(bank[['season', 'club_id'] + feature_cols],
                            on=['season', 'club_id'], how='left', validate='many_to_one')
    combined['feature_source'] = 'observed_final_profile'
    missing = combined[feature_cols].notna().sum(axis=1).eq(0)
    if missing.any() and not estimate_missing:
        names = combined.loc[missing, 'fifa_name'].tolist()
        raise ValueError('These clubs have no precomputed player profile: '
                         f'{names}. Provide an all-club feature bank or explicitly '
                         'set estimate_missing=True for an approximate sensitivity run.')

    # Backfill only explicitly permitted missing donor profiles, using complete
    # precomputed profiles with similar FIFA CLUB scores (not new Top XI).
    if missing.any():
        complete = bank.loc[bank[CLUB_RATINGS[:4]].notna().all(axis=1)]
        complete = complete.loc[complete[feature_cols].notna().any(axis=1)]
        current = complete.loc[complete.season.eq('2026-27')]
        if len(current) >= neighbor_k:
            complete = current
        if len(complete) < neighbor_k:
            raise ValueError('Not enough complete reference profiles for estimation')

        scales = np.array([8., 9., 9., 9.])
        matrix = complete[CLUB_RATINGS[:4]].to_numpy(float) / scales
        for idx in combined.index[missing]:
            donor = combined.loc[idx]
            point = donor[CLUB_RATINGS[:4]].to_numpy(float) / scales
            distances = np.linalg.norm(matrix - point, axis=1)
            picks = np.argsort(distances)[:neighbor_k]
            nearest = complete.iloc[picks]
            weights = 1 / (distances[picks] + 0.25)
            for feature in feature_cols:
                values = pd.to_numeric(nearest[feature], errors='coerce').to_numpy(float)
                valid = np.isfinite(values)
                if valid.any():
                    combined.loc[idx, feature] = np.average(values[valid], weights=weights[valid])
            combined.loc[idx, 'feature_source'] = 'estimated_from_rating_neighbors'
            combined.loc[idx, 'reference_clubs'] = '; '.join(nearest.club_name.astype(str))

    # Final models use attributes such as att_attacking_finishing_max, not
    # att1_attacking_finishing. Compute them per donor BEFORE weighted averaging.
    combined, aggregate_cols = make_position_statistics(combined)
    feature_cols = feature_cols + aggregate_cols
    combined['feature_missing_count'] = combined[feature_cols].isna().sum(axis=1)
    return combined, feature_cols


def make_position_statistics(donors):
    """Aggregate existing position slots per donor, before Elo/Golo averaging.

    Means commute with donor averaging, but min/max/std do not. Therefore
    statistics must first be computed for each donor team independently.
    """
    groups = {'gk': ['gk1'], 'def': [f'def{i}' for i in range(1, 5)],
              'mid': [f'mid{i}' for i in range(1, 4)],
              'att': [f'att{i}' for i in range(1, 4)]}
    result = {}

    for position, slots in groups.items():
        prefix = slots[0] + '_'
        attributes = [c[len(prefix):] for c in donors.columns if c.startswith(prefix)]

        for attr in attributes:
            columns = [f'{slot}_{attr}' for slot in slots]
            if not all(col in donors for col in columns):
                continue
            values = donors[columns].apply(pd.to_numeric, errors='coerce')
            stem = f'{position}_{attr}'
            result[f'{stem}_mean'] = values.mean(axis=1)
            if len(slots) > 1:
                result[f'{stem}_min'] = values.min(axis=1)
                result[f'{stem}_max'] = values.max(axis=1)
                result[f'{stem}_std'] = values.std(axis=1, ddof=0)

    if not result:
        raise ValueError('Feature bank contains no gk1/def1/mid1/att1 slot attributes')
    return pd.concat([donors, pd.DataFrame(result, index=donors.index)], axis=1), list(result)


def weighted_profile(neighbors, features):
    """Elo/Golo weighted mean of *existing* or explicitly estimated team features."""
    result = {}
    for col in features:
        values = pd.to_numeric(neighbors[col], errors='coerce')
        valid = values.notna()
        if valid.any():
            result[col] = float(np.average(values[valid], weights=neighbors.loc[valid, 'weight']))
    return result


def compare_proxies(elo_csv, clubs_csv, bank_csv, out_dir,
                    k_values=(5, 24), estimate_missing=False, allow_older=False):
    """Create both proxy tables and CSV exports; one call, one module."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    donors = load_donors(elo_csv, clubs_csv, allow_older=allow_older)
    donors, player_features = attach_final_features(
        donors, bank_csv, estimate_missing=estimate_missing
    )
    features = CLUB_RATINGS + player_features
    neighbors, records = [], []

    for target in TARGETS:
        for k in k_values:
            near = choose_neighbors(donors, target, k)
            neighbors.append(near)
            profile = weighted_profile(near, features)
            records.append({
                'team': target, 'k': k,
                'n_estimated_donors': int(near.feature_source.ne('observed_final_profile').sum()),
                'n_features': len(profile),
                'mean_elo_gap': float(np.average(abs(near.elo_gap), weights=near.weight)),
                'mean_golo_gap': float(np.average(abs(near.golo_gap), weights=near.weight)),
                **profile
            })

    neighbors = pd.concat(neighbors, ignore_index=True)
    profiles = pd.DataFrame(records)
    values = [c for c in features if c in profiles]
    wide = profiles.pivot(index='team', columns='k', values=values)
    k1, k2 = k_values
    deltas = pd.DataFrame({c: wide[c][k2] - wide[c][k1] for c in values}).reset_index()

    donors.to_csv(out_dir / 'matched_candidate_fifa_clubs.csv', index=False)
    neighbors.to_csv(out_dir / 'proxy_neighbors.csv', index=False)
    profiles.to_csv(out_dir / 'proxy_profiles.csv', index=False)
    deltas.to_csv(out_dir / 'proxy_profile_differences.csv', index=False)
    return neighbors, profiles, deltas


def build_partial_fixture_profiles(fixtures):
    """Extract existing FIFA attributes, allowing incomplete Sabah/Slovan."""
    attributes = sorted({c[5:] for c in fixtures if c.startswith('home_')} &
                        {c[5:] for c in fixtures if c.startswith('away_')})
    profiles = {}
    for team in sorted(set(fixtures.home_team) | set(fixtures.away_team)):
        values = {}
        for attr in attributes:
            pieces = []
            for side in ('home', 'away'):
                series = fixtures.loc[fixtures[f'{side}_team'].eq(team), f'{side}_{attr}']
                pieces.extend(series.dropna().tolist())
            values[attr] = pieces[0] if pieces else np.nan
        profiles[team] = values
    return profiles


def patch_profiles(base_profiles, proxy_profiles, k):
    """Add weighted profiles; do not alter fixtures or completed results."""
    patched = {team: dict(values) for team, values in base_profiles.items()}
    metadata = {'team', 'k', 'n_estimated_donors', 'n_features', 'mean_elo_gap', 'mean_golo_gap'}
    for row in proxy_profiles.loc[proxy_profiles.k.eq(k)].to_dict('records'):
        team = row['team']
        patched.setdefault(team, {})
        patched[team].update({c: v for c, v in row.items()
                              if c not in metadata and pd.notna(v)})
    return patched


def compare_monte_carlo(fixtures, base_profiles, proxy_profiles, final_models,
                        feature_groups, predict_bundle, runs=300, seed=42,
                        methods=('two_goals', 'soft_vote', 'goal_difference')):
    """Compare Top 8, Top 24, and champion probabilities under both proxy sizes."""
    import ucl_simulator as sim

    required = {c[5:] for bundle in final_models.values() for c in bundle['columns']
                if c.startswith(('home_', 'away_'))}
    required.difference_update({'team', 'club_id'})
    all_results, entries = {}, []
    for k in (5, 24):
        patched = patch_profiles(base_profiles, proxy_profiles, k)
        for target in TARGETS:
            missing = sorted(c for c in required if pd.isna(patched[target].get(c)))
            if missing:
                raise ValueError(f'{target}, Top {k}: missing model inputs: {missing[:10]}')
        cache = sim.make_prediction_cache(patched, final_models, feature_groups, predict_bundle)
        for method in methods:
            result = sim.run_monte_carlo(fixtures, cache, n=runs, method=method,
                                         seed=seed, tie_break='name')
            all_results[(k, method)] = result
            entries.append(result['odds'].assign(k=k, method=method))

    odds = pd.concat(entries, ignore_index=True)
    wide = odds.pivot(index=['team', 'method'], columns='k',
                      values=['top_8', 'top_24', 'champion'])
    wide.columns = [f'{metric}_k{k}' for metric, k in wide.columns]
    for metric in ('top_8', 'top_24', 'champion'):
        wide[f'{metric}_delta'] = wide[f'{metric}_k24'] - wide[f'{metric}_k5']
    return wide.reset_index(), all_results


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--elo', default='clubelo_screenshot_candidates.csv')
    p.add_argument('--clubs', default='data/output_final/clubs_2015_2027.csv')
    p.add_argument('--bank', default='data/match_dataset/team_feature_bank.csv')
    p.add_argument('--out', default='model/results/proxy_top5_top24')
    p.add_argument('--estimate-missing', action='store_true',
                   help='Explicitly estimate unavailable donor player profiles using FIFA ratings')
    p.add_argument('--allow-older', action='store_true')
    args = p.parse_args()
    nearest, profiles, differences = compare_proxies(
        args.elo, args.clubs, args.bank, args.out,
        estimate_missing=args.estimate_missing, allow_older=args.allow_older
    )
    print(profiles[['team', 'k', 'n_estimated_donors', 'n_features',
                    'club_overall', 'club_attack', 'club_midfield', 'club_defence']].round(3))
    print('Saved:', args.out)
