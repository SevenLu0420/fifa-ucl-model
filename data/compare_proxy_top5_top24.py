import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = {'Sabah': (1561.0, 1.49), 'Slovan Bratislava': (1540.0, 1.51)}

# Exact aliases checked against clubs_2015_2027.csv. No arbitrary fuzzy joins.
ALIASES = {
    'Cardiff': 'Cardiff City', 'Sirius': 'IK Sirius', 'Verona': 'Hellas Verona',
    'Valladolid': 'Real Valladolid', 'Molde': 'Molde FK',
    'Bristol City': 'Bristol City', 'Red Star': 'Red Star FC (FRANCE)',
    'Shanghai Port FC': 'Shanghai Port', 'Metz': 'FC Metz',
    'Mechelen': 'KV Mechelen', 'Erzurum': 'Erzurumspor FK',
    'Tromso': 'Tromsø IL', 'Servette FC': 'Servette FC',
    'Karlsruhe': 'Karlsruher SC', 'Annecy': 'FC Annecy',
    'Essen': 'Rot-Weiß Essen', 'Charlton': 'Charlton Athletic',
    'Duisburg': 'MSV Duisburg', 'Thun': 'FC Thun',
    'Everton Chile': 'Everton de Viña del Mar',
    'Nancy': 'AS Nancy', 'Lubin': 'Zagłębie Lubin',
    'Basel': 'FC Basel', 'Recoleta': 'Recoleta FC',
    'Lillestrom': 'Lillestrøm SK',
}

RATING_COLS = ['overall', 'attack', 'midfield', 'defence',
               'domestic_prestige', 'international_prestige']


def normalize(name):
    text = unicodedata.normalize('NFKD', str(name))
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9]', '', text.lower())


def attach_fifa_records(elo, clubs, allow_previous=True):
    """Match ClubElo names to FIFA records, preferring 2026-27 snapshots."""
    clubs = clubs.copy()
    clubs = clubs[clubs['gender'].eq(0)] if 'gender' in clubs else clubs
    clubs['_key'] = clubs['club_name'].map(normalize)
    clubs['_year'] = pd.to_numeric(clubs['season'].str[:4], errors='coerce')
    clubs = clubs.sort_values(['_year', 'snapshot_date'], ascending=False)

    matches = []
    for row in elo.itertuples(index=False):
        if row.team in TARGETS or row.team == 'Sabah FK':
            continue

        mapped = ALIASES.get(row.team, row.team)
        options = clubs[clubs['_key'].eq(normalize(mapped))]
        if not allow_previous:
            options = options[options.season.eq('2026-27')]
        if options.empty:
            continue
        found = options.iloc[0]
        rec = {'elo_name': row.team, 'fifa_name': found.club_name,
               'elo': float(row.elo), 'golo': float(row.golo),
               'season': found.season, 'club_id': found.club_id}
        rec.update({c: found[c] for c in RATING_COLS})
        matches.append(rec)

    matched = pd.DataFrame(matches)
    if matched.empty:
        raise ValueError('No candidate clubs matched the FIFA table')
    matched = matched.drop_duplicates('elo_name')
    return matched


def get_neighbors(matched, target, k, elo_scale=20, golo_scale=0.30,
                  recent_only=False):
    """Select k available FIFA clubs using combined scaled Elo/Golo distance."""
    target_elo, target_golo = TARGETS[target]
    pool = matched[matched.season.eq('2026-27')] if recent_only else matched
    pool = pool.copy()
    pool['elo_gap'] = pool.elo - target_elo
    pool['golo_gap'] = pool.golo - target_golo
    pool['distance'] = np.hypot(pool.elo_gap / elo_scale, pool.golo_gap / golo_scale)
    pool = pool.sort_values(['distance', 'elo_gap', 'elo_name'], key=None)

    if len(pool) < k:
        raise ValueError(f'{target}: only {len(pool)} matched FIFA candidates, requested {k}. '
                         'Add more same-day ClubElo rows or enable older FIFA seasons.')
    nearest = pool.head(k).copy()
    nearest['weight'] = (1 / (nearest.distance + 0.1))
    nearest['weight'] /= nearest.weight.sum()
    nearest['target'] = target
    nearest['k'] = k
    nearest['rank'] = np.arange(1, k + 1)
    return nearest


def weighted_profile(neighbors, feature_cols=RATING_COLS):
    """Weighted FIFA profile. Missing features are renormalized per attribute."""
    output = {}
    for col in feature_cols:
        if col not in neighbors.columns:
            continue
        values = pd.to_numeric(neighbors[col], errors='coerce')
        ok = values.notna()
        if ok.any():
            output[col] = float(np.average(values[ok], weights=neighbors.loc[ok, 'weight']))
    return output


def compare_proxies(elo_csv, clubs_csv, out_dir, k_values=(5, 24),
                    recent_only=True, feature_csv=None):
    """Write donor lists, weighted profiles, and the top-5 vs top-24 delta."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    elo = pd.read_csv(elo_csv)
    clubs = pd.read_csv(clubs_csv)
    donors = attach_fifa_records(elo, clubs, allow_previous=not recent_only)

    if feature_csv is not None:
        more = pd.read_csv(feature_csv)
        if not {'season', 'club_id'}.issubset(more.columns):
            raise ValueError('Additional feature file needs season and club_id')
        extra = [c for c in more.select_dtypes(include='number').columns
                 if c not in ['club_id', 'fifa_version'] and c not in RATING_COLS]
        donors = donors.merge(more[['season', 'club_id'] + extra],
                              on=['season', 'club_id'], how='left', validate='many_to_one')
    else:
        extra = []

    feature_cols = RATING_COLS + extra
    all_neighbors, rows = [], []
    for target in TARGETS:
        for k in k_values:
            near = get_neighbors(donors, target, k, recent_only=recent_only)
            all_neighbors.append(near)
            profile = weighted_profile(near, feature_cols)
            rows.append({'team': target, 'k': k, 'available_features': len(profile),
                         'fc27_donors': int(near.season.eq('2026-27').sum()),
                         'mean_elo_distance': float(np.average(abs(near.elo_gap), weights=near.weight)),
                         'mean_golo_distance': float(np.average(abs(near.golo_gap), weights=near.weight)),
                         **profile})

    neighbors = pd.concat(all_neighbors, ignore_index=True)
    profiles = pd.DataFrame(rows)
    metrics = [c for c in feature_cols if c in profiles]
    delta = profiles.pivot(index='team', columns='k', values=metrics)
    difference = pd.DataFrame({c: delta[(c, k_values[1])] - delta[(c, k_values[0])]
                               for c in metrics}, index=delta.index).reset_index()
    difference['metric'] = 'top24_minus_top5'

    neighbors.to_csv(out_dir / 'proxy_neighbors.csv', index=False)
    profiles.to_csv(out_dir / 'proxy_profiles.csv', index=False)
    difference.to_csv(out_dir / 'proxy_profile_differences.csv', index=False)
    donors.to_csv(out_dir / 'matched_candidate_fifa_clubs.csv', index=False)
    return neighbors, profiles, difference


def patch_profiles(profiles, proxy_profiles, method_k):
    """Replace only missing team profiles, without touching observed match scores."""
    result = {team: dict(values) for team, values in profiles.items()}
    for row in proxy_profiles[proxy_profiles.k.eq(method_k)].to_dict('records'):
        team = row.pop('team')
        row.pop('k', None)
        for key in ('available_features', 'fc27_donors', 'mean_elo_distance', 'mean_golo_distance'):
            row.pop(key, None)
        result.setdefault(team, {})
        result[team].update({('club_' + k if k in RATING_COLS else k): v
                             for k, v in row.items() if pd.notna(v)})
    return result




def build_partial_fixture_profiles(fixtures):
    """Extract known teams from fixtures, tolerating the two missing FIFA profiles."""
    attributes = sorted({c[5:] for c in fixtures if c.startswith('home_')} &
                        {c[5:] for c in fixtures if c.startswith('away_')})
    teams = sorted(set(fixtures.home_team) | set(fixtures.away_team))
    profiles = {}
    for team in teams:
        profile = {}
        for attr in attributes:
            values = []
            for side in ('home', 'away'):
                subset = fixtures.loc[fixtures[f'{side}_team'].eq(team), f'{side}_{attr}']
                values.extend(subset.dropna().tolist())
            profile[attr] = values[0] if values else np.nan
        profiles[team] = profile
    return profiles

def compare_monte_carlo(fixtures, base_profiles, proxy_profiles, final_models,
                        feature_groups, predict_bundle, runs=2000, seed=42,
                        methods=('two_goals', 'soft_vote', 'goal_difference')):
    """Optional: compare Top-8/24/title probabilities under k=5 vs k=24.

    Call from the final-model Notebook where the trained models are available.
    Use identical seeds so results have less Monte Carlo noise.
    """
    import ucl_simulator as sim
    results, rows = {}, []
    required_attrs = set()
    for bundle in final_models.values():
        for col in bundle['columns']:
            if col.startswith(('home_', 'away_')):
                required_attrs.add(col.split('_', 1)[1])
    required_attrs.difference_update({'team', 'club_id'})

    for k in (5, 24):
        patched = patch_profiles(base_profiles, proxy_profiles, k)
        for target in TARGETS:
            missing = sorted(c for c in required_attrs if pd.isna(patched[target].get(c)))
            if missing:
                raise ValueError(f'{target}, Top {k}: missing {len(missing)} model features '
                                 f'(examples: {missing[:6]}). Pass the full '
                                 'club_features_2015_2027.csv to compare_proxies().')
        cache = sim.make_prediction_cache(patched, final_models,
                                          feature_groups, predict_bundle)
        for method in methods:
            run = sim.run_monte_carlo(fixtures, cache, n=runs, method=method,
                                      seed=seed, tie_break='name')
            results[(k, method)] = run
            rows.append(run['odds'].assign(k=k, method=method))
    all_odds = pd.concat(rows, ignore_index=True)
    comparison = all_odds.pivot(index=['team', 'method'], columns='k',
                                values=['top_8', 'top_24', 'champion'])
    comparison.columns = [f'{name}_k{k}' for name, k in comparison.columns]
    for metric in ('top_8', 'top_24', 'champion'):
        comparison[f'{metric}_delta'] = comparison[f'{metric}_k24'] - comparison[f'{metric}_k5']
    return comparison.reset_index(), results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--elo', default='clubelo_screenshot_candidates.csv')
    parser.add_argument('--clubs', default='data/output_final/clubs_2015_2027.csv')
    parser.add_argument('--features', default=None,
                        help='Optional club_features_2015_2027.csv with position stats')
    parser.add_argument('--out', default='model/results/proxy_top5_top24')
    parser.add_argument('--allow-old', action='store_true',
                        help='Allow previous FIFA versions when FC27 is unavailable')
    args = parser.parse_args()

    neighbors, profiles, diff = compare_proxies(
        args.elo, args.clubs, args.out,
        recent_only=not args.allow_old, feature_csv=args.features
    )
    for team in TARGETS:
        print(f'\n{team} — nearest 5 and 24 FIFA donors')
        print(neighbors[(neighbors.target == team) & neighbors.k.eq(5)][
            ['fifa_name', 'season', 'elo', 'golo', 'weight']].to_string(index=False))
        print(profiles[profiles.team.eq(team)][
            ['k', 'overall', 'attack', 'midfield', 'defence',
             'fc27_donors', 'mean_elo_distance', 'mean_golo_distance']].round(3).to_string(index=False))
    print(f'\nSaved comparison to {args.out}')
