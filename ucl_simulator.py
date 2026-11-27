"""UCL 2026/27 conditional Monte Carlo simulator.

Uses the three fitted models from the final-model notebook. Match probabilities are
always in [away win, draw, home win] order. UEFA bracket structure follows
2026/27 Articles 17–21 and Annex B; see README in the companion notebook.

Important modeling assumptions:
- Two Goals predicts conditional means; independent Poisson generates scores.
- Soft Vote and Goal Difference change the H/D/A masses, while retaining the
  Two Goals conditional distribution of scores within each outcome.
- Extra time uses one third of 90-minute scoring rates; shoot-outs use 50/50
  unless explicitly replaced by a separately validated model.
- Inputs must be an as-of-date snapshot. Missing club profiles are never filled
  by unrelated teams or silently assigned made-up FIFA ratings.
"""

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam

CLASSES = ('A', 'D', 'H')
METHODS = ('two_goals', 'soft_vote', 'goal_difference')

# UEFA bracket: 9/10 v 23/24, 11/12 v 21/22, etc.
PLAYOFF_RANK_PAIRS = [((9, 10), (23, 24)), ((11, 12), (21, 22)),
                      ((13, 14), (19, 20)), ((15, 16), (17, 18))]

# Within each half, the R16 seed pairs occupy fixed quarters.
# seed (1/2) faces play-off group 3; (7/8) faces group 0;
# (3/4) faces group 2; (5/6) faces group 1.
R16_SLOT_ORDER = [(0, 3), (3, 0), (2, 2), (1, 1),
                  (0, 3), (3, 0), (2, 2), (1, 1)]
TOP_SEED_PAIRS = [(1, 2), (5, 6), (3, 4), (7, 8)]


def audit_fixtures(fixtures, as_of=None):
    """Check coverage and report whether simulation can run without invented data."""
    required = {'match_id', 'home_team', 'away_team', 'status',
                'label_90min_home_goals', 'label_90min_away_goals'}
    missing = required.difference(fixtures.columns)
    if missing:
        raise ValueError(f'Missing fixture columns: {sorted(missing)}')
    if fixtures['match_id'].duplicated().any():
        raise ValueError('Duplicate fixture match_id detected')

    teams = sorted(set(fixtures.home_team) | set(fixtures.away_team))
    result = {'matches': len(fixtures), 'teams': len(teams),
              'played': int(fixtures.status.eq('played').sum()),
              'scheduled': int(fixtures.status.eq('scheduled').sum())}
    bad = fixtures.loc[fixtures.status.eq('played') &
                       (fixtures.label_90min_home_goals.isna() |
                        fixtures.label_90min_away_goals.isna())]
    if len(bad):
        raise ValueError(f'{len(bad)} played fixtures have no 90-minute score')
    if as_of is not None and 'match_date' in fixtures:
        past_scheduled = fixtures.status.eq('scheduled') & (pd.to_datetime(fixtures.match_date).dt.date < pd.Timestamp(as_of).date())
        result['scheduled_in_past'] = int(past_scheduled.sum())
        if result['scheduled_in_past']:
            result['snapshot_warning'] = 'Scheduled matches precede as_of; update results before current forecast.'
    if 'prediction_ready' in fixtures:
        result['not_prediction_ready'] = int((~fixtures.prediction_ready.fillna(False).astype(bool)).sum())
        result['unready_teams'] = sorted(set(fixtures.loc[
            ~fixtures.prediction_ready.fillna(False).astype(bool), 'home_team']) |
            set(fixtures.loc[~fixtures.prediction_ready.fillna(False).astype(bool), 'away_team']))
    return result


def build_team_profiles(fixtures, required_columns=None):
    """Build season-level home/away-independent team feature dictionaries."""
    home_cols = [c for c in fixtures if c.startswith('home_')]
    away_cols = [c for c in fixtures if c.startswith('away_')]
    attributes = sorted(set(c[5:] for c in home_cols) & set(c[5:] for c in away_cols))
    profiles = {}
    for team in sorted(set(fixtures.home_team) | set(fixtures.away_team)):
        candidates = []
        for side in ('home', 'away'):
            subset = fixtures.loc[fixtures[f'{side}_team'].eq(team)]
            for _, row in subset.iterrows():
                values = {attr: row[f'{side}_{attr}'] for attr in attributes}
                candidates.append(values)
        profile = {}
        for attr in attributes:
            available = [r[attr] for r in candidates if pd.notna(r[attr])]
            profile[attr] = available[0] if available else np.nan
        profiles[team] = profile

    # Detect absent club information, not harmless player-level NaNs.
    checks = required_columns or ('club_overall', 'club_attack', 'club_midfield', 'club_defence')
    incomplete = [team for team, row in profiles.items()
                  if any(pd.isna(row.get(k)) for k in checks)]
    if incomplete:
        raise ValueError('Missing profiles for: ' + ', '.join(incomplete) +
                         '. Repair mapping/data first; do not silently impute clubs.')
    return profiles


def make_match_rows(pairs, profiles, stage='league', neutral=False):
    """Create fixture-like rows for arbitrary home/away pairs using same-season ratings."""
    rows = []
    for home, away in pairs:
        if home not in profiles or away not in profiles:
            raise KeyError(f'Missing team profile: {home} / {away}')
        row = {f'home_{k}': v for k, v in profiles[home].items()}
        row.update({f'away_{k}': v for k, v in profiles[away].items()})
        row.update({'home_team': home, 'away_team': away, 'neutral_venue': int(neutral),
                    'stage_league': int(stage == 'league'),
                    'stage_knockout': int(stage != 'league')})
        rows.append(row)
    return pd.DataFrame(rows)


def predict_matches(rows, saved, feature_groups, predict_bundle):
    """Evaluate three trained base models on arbitrary fixture rows, in batches."""
    groups = feature_groups(rows)
    output = {}
    for route in ('WDL', 'Goal Difference', 'Two Goals'):
        bundle = saved[route]
        X = groups[bundle['feature_set']].reindex(columns=bundle['columns'])
        X = X.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        probs, raw = predict_bundle(bundle, X)
        output[route] = (np.asarray(probs, dtype=float), raw)
    return output


def score_matrix(home_mu, away_mu, max_goals=12):
    """Independent-Poisson score distribution with explicit finite tail normalization."""
    values = np.arange(max_goals + 1)
    ph = poisson.pmf(values, max(0.01, float(home_mu)))
    pa = poisson.pmf(values, max(0.01, float(away_mu)))
    scores = np.outer(ph, pa)
    if scores.sum() < 0.999:
        raise ValueError('Score grid truncates material probability; raise max_goals')
    return scores / scores.sum()


def reweight_score_matrix(scores, target_probs):
    """Preserve P(score | outcome) while setting target P(A), P(D), P(H)."""
    i, j = np.indices(scores.shape)
    masks = [i < j, i == j, i > j]
    target_probs = np.asarray(target_probs, dtype=float)
    if not np.isclose(target_probs.sum(), 1, atol=1e-5):
        raise ValueError('Probabilities must sum to one')
    result = np.zeros_like(scores)
    for mask, target in zip(masks, target_probs):
        mass = scores[mask].sum()
        if mass <= 0:
            raise ValueError('No conditional score mass for outcome')
        result[mask] = scores[mask] * target / mass
    return result / result.sum()


def make_prediction_cache(profiles, saved, feature_groups, predict_bundle,
                          max_goals=12, symmetrize_neutral=True):
    """Precompute all pair, venue and engine probabilities for fast simulations.

    For a neutral final, averaging swapped home/away orientations reduces
    administrative 'home team' artifacts learned from neutral-venue history.
    """
    teams = sorted(profiles)
    pairs = [(h, a) for h in teams for a in teams if h != a]
    cache = {}
    for stage, neutral in [('league', False), ('knockout', False), ('final', True)]:
        rows = make_match_rows(pairs, profiles, stage=stage, neutral=neutral)
        predicted = predict_matches(rows, saved, feature_groups, predict_bundle)
        p_wdl = predicted['WDL'][0]
        p_gd = predicted['Goal Difference'][0]
        p_goals, means = predicted['Two Goals']
        home_means, away_means = means
        p_soft = (p_wdl + p_gd + p_goals) / 3

        for idx, pair in enumerate(pairs):
            base = score_matrix(home_means[idx], away_means[idx], max_goals)
            variants = {'two_goals': p_goals[idx], 'soft_vote': p_soft[idx],
                        'goal_difference': p_gd[idx]}
            for method, probs in variants.items():
                matrix = reweight_score_matrix(base, probs)
                cache[(stage, *pair, method)] = {'probs': probs.copy(), 'score': matrix}

    if symmetrize_neutral:
        for h, a in pairs:
            if h > a:
                continue
            for method in METHODS:
                key1 = ('final', h, a, method)
                key2 = ('final', a, h, method)
                m1 = cache[key1]['score']
                m2 = cache[key2]['score'].T
                combined = (m1 + m2) / 2
                cache[key1]['score'] = combined
                cache[key2]['score'] = combined.T
                masks = np.indices(combined.shape)
                p = np.array([combined[masks[0] < masks[1]].sum(),
                              combined[masks[0] == masks[1]].sum(),
                              combined[masks[0] > masks[1]].sum()])
                cache[key1]['probs'] = p
                cache[key2]['probs'] = p[::-1]

    # Precompute score CDF once for fast Monte Carlo draws.
    for obj in cache.values():
        obj['cdf'] = np.cumsum(obj['score'].ravel())
    return cache


def predict_match(cache, home, away, method='soft_vote', stage='league'):
    """Return P(A/D/H), most likely score and expected goals for one match."""
    obj = cache[(stage, home, away, method)]
    i, j = np.unravel_index(obj['score'].argmax(), obj['score'].shape)
    grid = np.indices(obj['score'].shape)
    return {'home': home, 'away': away, 'method': method,
            'p_away': obj['probs'][0], 'p_draw': obj['probs'][1],
            'p_home': obj['probs'][2], 'most_likely_score': f'{i}-{j}',
            'expected_home_goals': float((grid[0] * obj['score']).sum()),
            'expected_away_goals': float((grid[1] * obj['score']).sum())}


def simulate_match(cache, home, away, rng, method='soft_vote', stage='league'):
    """Sample a consistent 90-minute score and return an explanatory record."""
    obj = cache[(stage, home, away, method)]
    k = int(np.searchsorted(obj['cdf'], rng.random(), side='right'))
    gh, ga = np.unravel_index(min(k, obj['score'].size - 1), obj['score'].shape)
    label = 'H' if gh > ga else ('A' if gh < ga else 'D')
    return {'home': home, 'away': away, 'home_goals': int(gh), 'away_goals': int(ga),
            'result': label, 'p_home': float(obj['probs'][2]),
            'p_draw': float(obj['probs'][1]), 'p_away': float(obj['probs'][0]),
            'score_probability': float(obj['score'][gh, ga])}


def simulate_league_phase(fixtures, cache, rng, method='soft_vote'):
    """Keep completed scores and sample only future league-phase fixtures."""
    matches = []
    for _, row in fixtures.iterrows():
        if row.status == 'played':
            gh, ga = int(row.label_90min_home_goals), int(row.label_90min_away_goals)
            record = {'home': row.home_team, 'away': row.away_team,
                      'home_goals': gh, 'away_goals': ga,
                      'result': 'H' if gh > ga else ('A' if gh < ga else 'D'),
                      'status': 'played', 'match_id': row.match_id}
        elif row.status == 'scheduled':
            record = simulate_match(cache, row.home_team, row.away_team, rng, method)
            record.update({'status': 'simulated', 'match_id': row.match_id})
        else:
            raise ValueError(f'Unexpected status: {row.status}')
        matches.append(record)
    return pd.DataFrame(matches)


def rank_league_table(matches, all_teams=None, tie_break='error',
                      discipline=None, coefficient=None):
    """Rank league teams using UEFA rules, including opponents' combined records.

    If teams remain tied after available sporting criteria, the official rules
    require discipline and club coefficients. Pass those dictionaries to resolve
    exact ties, or set tie_break='name' for a clearly marked approximation.
    """
    teams = sorted(all_teams or (set(matches.home) | set(matches.away)))
    stats = {t: {'team': t, 'played': 0, 'points': 0, 'gd': 0, 'gf': 0,
                 'ga': 0, 'away_goals': 0, 'wins': 0, 'away_wins': 0} for t in teams}
    opponents = {t: [] for t in teams}
    for r in matches.itertuples(index=False):
        h, a, gh, ga = r.home, r.away, int(r.home_goals), int(r.away_goals)
        hs, ass = stats[h], stats[a]
        hs['played'] += 1; ass['played'] += 1
        hs['gf'] += gh; hs['ga'] += ga; ass['gf'] += ga; ass['ga'] += gh
        hs['away_goals'] += 0; ass['away_goals'] += ga
        hs['gd'] += gh - ga; ass['gd'] += ga - gh
        hs['points'] += 3 if gh > ga else (1 if gh == ga else 0)
        ass['points'] += 3 if ga > gh else (1 if gh == ga else 0)
        hs['wins'] += gh > ga; ass['wins'] += ga > gh
        ass['away_wins'] += ga > gh
        opponents[h].append(a); opponents[a].append(h)

    for team, s in stats.items():
        s['opp_points'] = sum(stats[o]['points'] for o in opponents[team])
        s['opp_gd'] = sum(stats[o]['gd'] for o in opponents[team])
        s['opp_gf'] = sum(stats[o]['gf'] for o in opponents[team])
        s['discipline'] = (discipline or {}).get(team, np.nan)
        s['coefficient'] = (coefficient or {}).get(team, np.nan)

    table = pd.DataFrame(stats.values())
    cols = ['points', 'gd', 'gf', 'away_goals', 'wins', 'away_wins',
            'opp_points', 'opp_gd', 'opp_gf']
    table = table.sort_values(cols + ['team'], ascending=[False] * len(cols) + [True])

    tied = table.duplicated(cols, keep=False)
    if tied.any():
        if table.loc[tied, ['discipline', 'coefficient']].isna().any().any():
            if tie_break != 'name':
                raise ValueError('UEFA tie requires discipline/coefficient for: ' +
                                 ', '.join(table.loc[tied, 'team']))
        else:
            cols += ['discipline', 'coefficient']
            table = table.sort_values(cols + ['team'],
                ascending=[False] * 9 + [True, False, True])
    return table.reset_index(drop=True).assign(rank=lambda d: np.arange(1, len(d) + 1))


def generate_valid_draws(rng=None, enumerate_all=False):
    """Eight binary UEFA draw choices: four play-off slots + four R16 seed slots.

    Return all 256 legally paired bracket assignments, or draw one uniformly.
    The tuple holds four play-off flips and four R16 pair flips.
    """
    if enumerate_all:
        return list(product((0, 1), repeat=8))
    if rng is None:
        rng = np.random.default_rng()
    return tuple(int(v) for v in rng.integers(0, 2, size=8))


def bracket_from_ranks(table, draw_bits):
    """Generate play-off ties and fixed R16 slots using ranks 1–24."""
    if len(draw_bits) != 8:
        raise ValueError('A draw must contain eight 0/1 choices')
    ranks = dict(zip(table['rank'], table['team']))
    playoff = []
    for i, (seeded, unseeded) in enumerate(PLAYOFF_RANK_PAIRS):
        u = unseeded if draw_bits[i] == 0 else unseeded[::-1]
        playoff.extend(({'seed': ranks[s], 'opponent': ranks[t],
                         'seed_rank': s, 'slot': i * 2 + j, 'group': i})
                       for j, (s, t) in enumerate(zip(seeded, u)))
    seed_pairs = [p if draw_bits[4 + i] == 0 else p[::-1]
                  for i, p in enumerate(TOP_SEED_PAIRS)]
    return playoff, seed_pairs


def _et_and_penalties(home, away, cache, rng, method, regular_home, regular_away):
    """Resolve an aggregate tie after 90 minutes; never use away-goals rule."""
    info = predict_match(cache, home, away, method, stage='knockout')
    eh = rng.poisson(info['expected_home_goals'] / 3)
    ea = rng.poisson(info['expected_away_goals'] / 3)
    # This helper is called only when the aggregate score is tied.
    # First-leg goals cancel out; extra-time goals alone decide the tie.
    if eh != ea:
        return (home if eh > ea else away,
                int(eh), int(ea), False)
    return (home if rng.random() < 0.5 else away, int(eh), int(ea), True)


def simulate_two_leg_tie(cache, first_leg_home, second_leg_home, rng,
                         method='soft_vote', stage='playoff'):
    """Simulate two legs, then extra time and penalties at the second-leg venue."""
    a = simulate_match(cache, first_leg_home, second_leg_home, rng,
                       method, stage='knockout')
    b = simulate_match(cache, second_leg_home, first_leg_home, rng,
                       method, stage='knockout')
    aggregate = {first_leg_home: a['home_goals'] + b['away_goals'],
                 second_leg_home: a['away_goals'] + b['home_goals']}
    extra = (0, 0); shootout = False
    if aggregate[first_leg_home] == aggregate[second_leg_home]:
        winner, eh, ea, shootout = _et_and_penalties(second_leg_home,
            first_leg_home, cache, rng, method, b['home_goals'], b['away_goals'])
        extra = (eh, ea)
    else:
        winner = max(aggregate, key=aggregate.get)
    return {'stage': stage, 'first_leg': a, 'second_leg': b, 'aggregate': aggregate,
            'extra_time': extra, 'penalties': shootout, 'winner': winner,
            'loser': second_leg_home if winner == first_leg_home else first_leg_home}



def estimate_tie_win_probability(cache, first_leg_home, second_leg_home,
                                 method='soft_vote', n=2000, seed=123):
    """Estimate advancement probability for the first-leg home team.

    This includes both legs, approximate extra time and shoot-outs. It does not
    account for dynamic injuries, suspension, or lineup changes between legs.
    """
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(n):
        tie = simulate_two_leg_tie(cache, first_leg_home, second_leg_home,
                                   rng, method=method)
        wins += tie['winner'] == first_leg_home
    p = wins / n
    se = np.sqrt(p * (1 - p) / n)
    return {'team': first_leg_home, 'opponent': second_leg_home,
            'win_probability': p, 'mc_ci_low': max(0, p - 1.96 * se),
            'mc_ci_high': min(1, p + 1.96 * se), 'runs': n}

def simulate_knockout(table, cache, rng, method='soft_vote', draw_bits=None):
    """Simulate play-offs, R16, quarters, semis and neutral final along fixed path."""
    bits = generate_valid_draws(rng) if draw_bits is None else tuple(draw_bits)
    playoff, seed_pairs = bracket_from_ranks(table, bits)
    matches = []
    winners_by_group = [[] for _ in range(4)]

    # Seeds 9–16 host their second legs.
    for t in playoff:
        result = simulate_two_leg_tie(cache, t['opponent'], t['seed'], rng,
                                     method, stage='playoff')
        result['slot'] = t['slot']
        winners_by_group[t['group']].append(result['winner'])
        matches.append(result)

    # R16 seed draw: each seed pair has one club on each half of the bracket.
    r16 = []
    for half in (0, 1):
        for pair_id, play_group in R16_SLOT_ORDER[half * 4:(half + 1) * 4]:
            higher = seed_pairs[pair_id][half]
            r16.append((table.iloc[higher - 1]['team'], winners_by_group[play_group][half], higher))

    bracket = []
    for i, (seeded, other, seed_rank) in enumerate(r16):
        result = simulate_two_leg_tie(cache, other, seeded, rng, method, stage='r16')
        result.update({'slot': i, 'path_priority': seed_rank})
        matches.append(result)
        bracket.append((result['winner'], seed_rank))

    # Higher seed's inherited bracket position determines preferred return venue.
    for stage in ('quarter_final', 'semi_final'):
        next_round = []
        for i in range(0, len(bracket), 2):
            (one, p1), (two, p2) = bracket[i], bracket[i + 1]
            preferred = one if p1 < p2 else two
            other = two if preferred == one else one
            result = simulate_two_leg_tie(cache, other, preferred, rng, method, stage)
            result.update({'slot': i // 2, 'path_priority': min(p1, p2)})
            matches.append(result)
            next_round.append((result['winner'], min(p1, p2)))
        bracket = next_round

    silver, blue = bracket[0][0], bracket[1][0]
    final = simulate_match(cache, silver, blue, rng, method, stage='final')
    shootout = False; extra = (0, 0)
    if final['home_goals'] == final['away_goals']:
        winner, eh, ea, shootout = _et_and_penalties(silver, blue, cache, rng,
            method, final['home_goals'], final['away_goals'])
        extra = (eh, ea)
    else:
        winner = silver if final['home_goals'] > final['away_goals'] else blue
    matches.append({'stage': 'final', 'first_leg': final, 'winner': winner,
                    'extra_time': extra, 'penalties': shootout, 'slot': 0})
    return winner, matches, bits


def simulate_tournament(fixtures, cache, rng, method='soft_vote', tie_break='name'):
    """Return a single league table, completed bracket and champion."""
    league = simulate_league_phase(fixtures, cache, rng, method)
    table = rank_league_table(league, tie_break=tie_break)
    champion, knockout, draw = simulate_knockout(table, cache, rng, method)
    return {'champion': champion, 'table': table, 'league_matches': league,
            'knockout': knockout, 'draw_bits': draw}


def run_monte_carlo(fixtures, cache, n=10000, method='soft_vote', seed=42,
                    tie_break='name', representative=True,
                    bracket_weights=(0.5, 0.3, 0.2)):
    """One tournament loop: return advance odds and a representative bracket.

    If representative=True, each simulated knockout bracket is packed into a
    compact int16 array; NO second tournament sampling loop is needed.
    """
    rng = np.random.default_rng(seed)
    teams = sorted(set(fixtures.home_team) | set(fixtures.away_team))
    stages = ['top_8', 'top_24', 'r16', 'quarter_final', 'semi_final', 'final', 'champion']
    counts = {t: {s: 0 for s in stages} for t in teams}
    table_records = []
    example = None
    # Each row has 23 ties: (first-leg home, first-leg away, winner), IDs.
    # ~2.8 MB for 20,000 runs, rather than storing full tournament objects.
    team_to_id = {team: i for i, team in enumerate(teams)}
    compact_paths = (np.empty((n, 23, 3), dtype=np.int16)
                     if representative else None)
    champion_ids = (np.empty(n, dtype=np.int16) if representative else None)
    for run in range(n):
        outcome = simulate_tournament(fixtures, cache, rng, method, tie_break)
        if example is None:
            example = outcome
        if representative:
            for tie_index, tie in enumerate(outcome['knockout']):
                first, second = _tie_participants(tie)
                compact_paths[run, tie_index] = (team_to_id[first],
                    team_to_id[second], team_to_id[tie['winner']])
            champion_ids[run] = team_to_id[outcome['champion']]
        table = outcome['table']
        for row in table.itertuples(index=False):
            counts[row.team]['top_8'] += row.rank <= 8
            counts[row.team]['top_24'] += row.rank <= 24
            table_records.append((row.team, row.rank, row.points, row.gd))
        for tie in outcome['knockout']:
            stage = tie['stage']
            if stage == 'playoff':
                counts[tie['winner']]['r16'] += 1
            elif stage == 'r16':
                counts[tie['winner']]['quarter_final'] += 1
            elif stage == 'quarter_final':
                counts[tie['winner']]['semi_final'] += 1
            elif stage == 'semi_final':
                counts[tie['winner']]['final'] += 1
        for t in table.head(8)['team']:
            counts[t]['r16'] += 1
        counts[outcome['champion']]['champion'] += 1

    odds = pd.DataFrame([{'team': t, **{s: counts[t][s] / n for s in stages}}
                         for t in teams]).sort_values('champion', ascending=False)
    odds['mc_se'] = np.sqrt(odds.champion * (1 - odds.champion) / n)
    odds['mc_ci_low'] = np.maximum(0, odds.champion - 1.96 * odds.mc_se)
    odds['mc_ci_high'] = np.minimum(1, odds.champion + 1.96 * odds.mc_se)
    ranks = pd.DataFrame(table_records, columns=['team', 'rank', 'points', 'gd'])
    league_summary = ranks.groupby('team').agg(
        mean_rank=('rank', 'mean'), mean_points=('points', 'mean'),
        mean_goal_diff=('gd', 'mean'), rank_p10=('rank', lambda x: x.quantile(0.1)),
        rank_p90=('rank', lambda x: x.quantile(0.9))).reset_index()
    result = {'method': method, 'runs': n, 'odds': odds.reset_index(drop=True),
              'league_summary': league_summary, 'sample': example}
    if representative:
        result['representative'] = _select_representative_from_compact(
            compact_paths, champion_ids, teams, result['odds'], cache,
            method, weights=bracket_weights)
    return result


def plot_championship_odds(results, top=15):
    """Plot champion probability with 95% Monte Carlo sampling-error intervals."""
    d = results['odds'].head(top).sort_values('champion')
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * top + 1)))
    ax.barh(d.team, d.champion * 100)
    ax.errorbar(d.champion * 100, d.team, xerr=1.96 * d.mc_se * 100,
                fmt='none', color='black', capsize=2)
    ax.set_xlabel('Championship probability (%)')
    ax.set_title(f"UCL title probabilities — {results['method']} ({results['runs']:,} simulations)")
    fig.tight_layout()
    return fig


def plot_league_table(results, top=20):
    """Plot expected final league positions and 10th–90th percentile bands."""
    d = results['league_summary'].nsmallest(top, 'mean_rank').sort_values('mean_rank', ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * top + 1)))
    ax.errorbar(d.mean_rank, d.team,
                xerr=[d.mean_rank - d.rank_p10, d.rank_p90 - d.mean_rank],
                fmt='o', capsize=3)
    ax.axvline(8.5, linestyle='--', color='gray', linewidth=1)
    ax.axvline(24.5, linestyle=':', color='gray', linewidth=1)
    ax.set_xlabel('Final league rank (10th–90th percentile)')
    ax.set_title('Simulated league table: mean ranks and uncertainty')
    ax.invert_xaxis()
    fig.tight_layout()
    return fig


def plot_bracket(sample):
    """Plot one representative simulated knockout bracket by stage and winner."""
    stages = [('playoff', 'Play-offs'), ('r16', 'R16'),
              ('quarter_final', 'QF'), ('semi_final', 'SF'), ('final', 'Final')]
    fig, axes = plt.subplots(1, 5, figsize=(19, 9), sharey=True)
    for ax, (stage, title) in zip(axes, stages):
        ties = sorted((r for r in sample['knockout'] if r['stage'] == stage),
                      key=lambda r: r.get('slot', 0))
        for j, tie in enumerate(ties):
            if stage == 'final':
                record = tie['first_leg']
                label = f"{record['home']} {record['home_goals']}–{record['away_goals']} {record['away']}\nWIN: {tie['winner']}"
            else:
                names = list(tie['aggregate'])
                scores = tie['aggregate']
                decider = ' (pens)' if tie['penalties'] else (' (ET)' if tie['extra_time'] != (0, 0) else '')
                label = f"{names[0]} {scores[names[0]]}–{scores[names[1]]} {names[1]}\nWIN: {tie['winner']}{decider}"
            pos = 0.95 - (j + 0.5) / max(len(ties), 1) * 0.9
            ax.text(0.01, pos, label, ha='left', va='center', fontsize=7,
                    transform=ax.transAxes, bbox=dict(facecolor='none', edgecolor='gray', pad=3))
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle('One simulated UEFA knockout path (not the aggregate probability tree)')
    fig.tight_layout()
    return fig


def compare_model_odds(results_by_method, top=12):
    """Plot three alternative-model title forecasts side by side."""
    all_odds = pd.concat([r['odds'][['team', 'champion']].assign(method=method)
                          for method, r in results_by_method.items()])
    teams = all_odds.groupby('team').champion.mean().nlargest(top).index
    chart = all_odds[all_odds.team.isin(teams)].pivot(index='team', columns='method', values='champion')
    ax = (100 * chart).plot(kind='barh', figsize=(11, max(5, 0.4 * top)))
    ax.set_xlabel('Championship probability (%)')
    ax.set_title('Model sensitivity: Two Goals vs Soft Vote vs Goal Difference')
    ax.figure.tight_layout()
    return ax.figure

# === REPRESENTATIVE KNOCKOUT BRACKET ===
# These functions extend the original tournament simulator. They do not modify
# saved models, input fixtures, or the tournament rules used above.


def _tie_participants(tie):
    """Return the two teams in the original first-leg/final display order."""
    first = tie['first_leg']
    return first['home'], first['away']


def _conditional_extra_time_probability(cache, second_home, second_away, method):
    """P(second-home team advances | two-leg aggregate tied after 90 minutes).

    The simulator samples extra-time scores as independent Poisson variables
    with one third of the predicted 90-minute goals and assumes 50:50 penalties.
    """
    info = predict_match(cache, second_home, second_away,
                         method=method, stage='knockout')
    mu_home = info['expected_home_goals'] / 3.0
    mu_away = info['expected_away_goals'] / 3.0
    home_wins_et = 1.0 - skellam.cdf(0, mu_home, mu_away)
    tied_et = skellam.pmf(0, mu_home, mu_away)
    return float(np.clip(home_wins_et + 0.5 * tied_et, 0.0, 1.0))


def _two_leg_probability_and_mode(cache, first_home, first_away,
                                  winner, method):
    """Exact advancement probability under the simulator's truncated scores.

    Also returns the most probable *pair of regulation-time leg scores* among
    outcomes in which the specified winner advances. A regulation-time aggregate
    tie is weighted by its extra-time/penalty advancement probability.
    """
    if winner not in (first_home, first_away):
        raise ValueError(f'{winner} does not participate in this tie')

    first = cache[('knockout', first_home, first_away, method)]['score']
    second = cache[('knockout', first_away, first_home, method)]['score']
    if first.shape[0] != first.shape[1] or second.shape != first.shape:
        raise ValueError('Expected square, equally sized score matrices')

    max_goal = first.shape[0] - 1
    i, j = np.indices(first.shape)
    d_first = i - j
    d_second = i - j  # second leg: first_away at home
    support = 2 * max_goal + 1
    p_d1 = np.bincount((d_first + max_goal).ravel(),
                       weights=first.ravel(), minlength=support)
    p_d2 = np.bincount((d_second + max_goal).ravel(),
                       weights=second.ravel(), minlength=support)
    aggregate_d = np.convolve(p_d1, p_d2[::-1])
    zero = 2 * max_goal

    p_first_home_if_tied = 1.0 - _conditional_extra_time_probability(
        cache, first_away, first_home, method)
    p_first_home = float(aggregate_d[zero + 1:].sum()
                         + aggregate_d[zero] * p_first_home_if_tied)
    p_winner = p_first_home if winner == first_home else 1.0 - p_first_home

    # Joint score probability for first leg (first_home-first_away) and
    # second leg (first_away-first_home). This is 13^4 entries at max_goals=12.
    joint = first[:, :, None, None] * second[None, None, :, :]
    diff = d_first[:, :, None, None] - d_second[None, None, :, :]
    tie_factor = (p_first_home_if_tied if winner == first_home
                  else 1.0 - p_first_home_if_tied)
    admissible = ((diff > 0) if winner == first_home else (diff < 0))
    weighted = joint * (admissible + (diff == 0) * tie_factor)
    idx = np.unravel_index(int(np.argmax(weighted)), weighted.shape)
    a1, b1, b2, a2 = (int(v) for v in idx)
    agg_a, agg_b = a1 + a2, b1 + b2
    is_tied = agg_a == agg_b
    description = f'{agg_a}–{agg_b} agg.' + (' · ET/PEN' if is_tied else '')

    return {
        'advance_probability': float(np.clip(p_winner, 0.0, 1.0)),
        'reference_score': description,
        'reference_first_leg': f'{a1}–{b1}',
        'reference_second_leg': f'{b2}–{a2}',
        'reference_aggregate': f'{agg_a}–{agg_b}',
        'reference_needs_tiebreak': bool(is_tied),
        'reference_joint_probability': float(weighted[idx]),
        'reference_conditional_probability': float(weighted[idx] / max(p_winner, 1e-12)),
    }


def _final_probability_and_mode(cache, home, away, winner, method):
    """One-leg final; 90-minute draw is resolved by ET/50:50 penalties."""
    if winner not in (home, away):
        raise ValueError(f'{winner} does not participate in the final')
    mat = cache[('final', home, away, method)]['score']
    gh, ga = np.indices(mat.shape)
    info = predict_match(cache, home, away, method=method, stage='final')
    mu_h = info['expected_home_goals'] / 3.0
    mu_a = info['expected_away_goals'] / 3.0
    p_home_if_tied = float(
        1 - skellam.cdf(0, mu_h, mu_a) + 0.5 * skellam.pmf(0, mu_h, mu_a))
    weights = mat * ((gh > ga) + (gh == ga) * p_home_if_tied)
    p_home = float(weights.sum())
    if winner != home:
        weights = mat * ((gh < ga) + (gh == ga) * (1 - p_home_if_tied))
    p_winner = float(weights.sum())
    r, c = (int(v) for v in np.unravel_index(np.argmax(weights), weights.shape))
    is_tied = (r == c)
    return {
        'advance_probability': p_winner,
        'reference_score': f"{r}–{c} (90')" + (' · ET/PEN' if is_tied else ''),
        'reference_first_leg': f'{r}–{c}',
        'reference_second_leg': '',
        'reference_aggregate': f'{r}–{c}',
        'reference_needs_tiebreak': bool(is_tied),
        'reference_joint_probability': float(weights[r, c]),
        'reference_conditional_probability': float(weights[r, c] / max(p_winner, 1e-12)),
    }


def _tie_reference(cache, tie, method, memo):
    """Memoize analytical advancement odds and consistent conditional scores."""
    a, b = _tie_participants(tie)
    winner = tie['winner']
    stage = tie['stage']
    key = (method, stage, a, b, winner)
    if key not in memo:
        if stage == 'final':
            memo[key] = _final_probability_and_mode(cache, a, b, winner, method)
        else:
            memo[key] = _two_leg_probability_and_mode(cache, a, b, winner, method)
    return memo[key]


def _round_entrants(outcome):
    """Teams actually appearing in each round, derived from a valid full path."""
    stage_names = ('playoff', 'r16', 'quarter_final', 'semi_final', 'final')
    entrants = {stage: set() for stage in stage_names}
    for tie in outcome['knockout']:
        entrants[tie['stage']].update(_tie_participants(tie))
    assert len(entrants['playoff']) == 16
    assert len(entrants['r16']) == 16
    assert len(entrants['quarter_final']) == 8
    assert len(entrants['semi_final']) == 4
    assert len(entrants['final']) == 2
    return entrants


# The round boundaries correspond to simulate_knockout() order (8+8+4+2+1).
_COMPACT_ROUNDS = (('playoff', 0, 8), ('r16', 8, 16),
                   ('quarter_final', 16, 20), ('semi_final', 20, 22),
                   ('final', 22, 23))


def _fast_advance_probability(cache, a, b, winner, method, stage, memo):
    """Advancement probability only; avoids expensive score-mode enumeration.

    Teams a,b are first-leg home/away, respectively. The second leg is b-a.
    The final has one neutral 90-minute match.
    """
    key = (stage, a, b, method)
    if key not in memo:
        if stage == 'final':
            mat = cache[('final', a, b, method)]['score']
            i, j = np.indices(mat.shape)
            p_et = _conditional_extra_time_probability_final(cache, a, b, method)
            p_a = float(mat[i > j].sum() + p_et * mat[i == j].sum())
        else:
            first = cache[('knockout', a, b, method)]['score']
            second = cache[('knockout', b, a, method)]['score']
            max_goals = first.shape[0] - 1
            ii, jj = np.indices(first.shape)
            d = ii - jj
            p1 = np.bincount((d + max_goals).ravel(),
                             weights=first.ravel(), minlength=2*max_goals+1)
            p2 = np.bincount((d + max_goals).ravel(),
                             weights=second.ravel(), minlength=2*max_goals+1)
            agg = np.convolve(p1, p2[::-1])
            zero = 2 * max_goals
            p_et_b = _conditional_extra_time_probability(cache, b, a, method)
            p_a = float(agg[zero+1:].sum() + agg[zero] * (1-p_et_b))
        memo[key] = min(1.0, max(0.0, p_a))
    return memo[key] if winner == a else 1.0 - memo[key]


def _conditional_extra_time_probability_final(cache, a, b, method):
    info = predict_match(cache, a, b, method=method, stage='final')
    h, v = info['expected_home_goals'] / 3, info['expected_away_goals'] / 3
    return float(1 - skellam.cdf(0, h, v) + 0.5*skellam.pmf(0, h, v))


def _restore_compact_bracket(compact, teams, champion):
    """Restore the exact sampled matchup/winner path for plotting only.

    Scores are intentionally not stored: plot functions compute modal
    conditional reference scores for the chosen matchups.
    """
    ties = []
    for stage, lo, hi in _COMPACT_ROUNDS:
        for slot, (a, b, w) in enumerate(compact[lo:hi]):
            team_a, team_b, winner = teams[int(a)], teams[int(b)], teams[int(w)]
            tie = {'stage': stage, 'slot': slot,
                   'first_leg': {'home': team_a, 'away': team_b},
                   'winner': winner}
            ties.append(tie)
    if ties[-1]['winner'] != champion:
        raise AssertionError('Stored bracket champion mismatch')
    return {'champion': champion, 'knockout': ties}


def _select_representative_from_compact(paths, champions, teams, odds, cache,
                                         method, weights=(0.5, 0.3, 0.2)):
    """Pick a champion-conditional knockout medoid from SAME Monte Carlo runs."""
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (3,) or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError('bracket_weights must be three nonnegative values summing to one')
    best_champion = str(odds.sort_values('champion', ascending=False).iloc[0]['team'])
    champion_id = teams.index(best_champion)
    eligible = np.flatnonzero(champions == champion_id)
    if not len(eligible):
        raise ValueError('No simulated paths for top-probability champion')
    subset = paths[eligible]
    # Conditional round-presence probabilities per team, per round.
    freqs = {}
    rep_sum = np.zeros(len(subset), dtype=float)
    rep_terms = 0
    for stage, lo, hi in _COMPACT_ROUNDS:
        # Each team appears at most once per round.
        entrants = np.concatenate((subset[:,lo:hi,0], subset[:,lo:hi,1]), axis=1)
        presence = np.bincount(entrants.ravel().astype(int),
                               minlength=len(teams)) / len(subset)
        freqs[stage] = {teams[i]: float(p) for i,p in enumerate(presence) if p > 0}
        rep_sum += np.log(np.maximum(presence[entrants], 1e-12)).sum(axis=1)
        rep_terms += entrants.shape[1]
    representation = np.exp(rep_sum / rep_terms)
    # Compute each distinct pairing once; leverage analytic advancement odds.
    memo = {}
    a = subset[:,:,0].ravel().astype(int)
    b = subset[:,:,1].ravel().astype(int)
    w = subset[:,:,2].ravel().astype(int)
    stage_code = np.broadcast_to(np.repeat(np.arange(5), [8,8,4,2,1]),
                                 subset.shape[:2]).ravel()
    unique_keys, inverse = np.unique(
        np.stack([stage_code,a,b,w],axis=1),axis=0,return_inverse=True)
    probs_unique = np.empty(len(unique_keys), dtype=float)
    for i, (stage_id, ia, ib, iw) in enumerate(unique_keys):
        stage = _COMPACT_ROUNDS[int(stage_id)][0]
        probs_unique[i] = _fast_advance_probability(
            cache, teams[int(ia)],teams[int(ib)],teams[int(iw)],method,stage,memo)
    p = np.clip(probs_unique[inverse].reshape(len(subset),23), 1e-12, 1.0)
    confidence = p.mean(axis=1)
    likelihood = np.exp(np.log(p).mean(axis=1))
    score = weights[0]*representation + weights[1]*confidence + weights[2]*likelihood
    best_local = int(np.argmax(score))
    best_run = int(eligible[best_local])
    selected = _restore_compact_bracket(paths[best_run], teams, best_champion)
    selected_scores = pd.DataFrame({
        'run_index': eligible, 'score': score, 'representation': representation,
        'match_confidence': confidence, 'path_likelihood_geomean': likelihood
    }).sort_values(['score','run_index'],ascending=[False,True]).reset_index(drop=True)
    return {'champion': best_champion, 'method': method, 'path': selected,
            'n_simulations': int(len(paths)), 'n_champion_paths': int(len(eligible)),
            'weights': tuple(weights), 'candidate_scores': selected_scores,
            'conditional_round_probabilities': freqs,
            'selected_score': float(score[best_local]),
            'selected_run_index': best_run}


def representative_bracket_table(selection, cache):
    """One row per tie, with fixture-specific advancement odds and score modes."""
    method = selection['method']
    memo = {}
    rows = []
    for tie in selection['path']['knockout']:
        first, second = _tie_participants(tie)
        ref = _tie_reference(cache, tie, method, memo)
        rows.append({'stage': tie['stage'], 'slot': tie['slot'],
                     'team_1': first, 'team_2': second, 'winner': tie['winner'],
                     'winner_probability': ref['advance_probability'],
                     'reference_score': ref['reference_score'],
                     'first_leg_reference': ref['reference_first_leg'],
                     'second_leg_reference': ref['reference_second_leg'],
                     'reference_conditional_probability':
                         ref['reference_conditional_probability']})
    stage_order = {'playoff': 0, 'r16': 1, 'quarter_final': 2,
                   'semi_final': 3, 'final': 4}
    table = pd.DataFrame(rows)
    table['_stage_order'] = table.stage.map(stage_order)
    return table.sort_values(['_stage_order', 'slot']).drop(
        columns='_stage_order').reset_index(drop=True)


def plot_representative_knockout_bracket(selection, cache, championship_odds=None,
                                         figsize=(22, 12)):
    """Draw one clean, connected, 23-tie bracket, plus the predicted champion.

    Probabilities next to winners are **unconditional to the chosen champion**:
    they mean P(that team advances | these two teams meet with these venues).
    Reference scores are modal regulation-time score pairs conditional on the
    selected winner, not scores sampled in the representative Monte Carlo path.
    """
    from matplotlib.patches import FancyBboxPatch

    path = selection['path']
    method = selection['method']
    table = representative_bracket_table(selection, cache)
    ties = {s: sorted((tie for tie in path['knockout'] if tie['stage'] == s),
                       key=lambda t: t['slot'])
            for s in ('playoff', 'r16', 'quarter_final', 'semi_final', 'final')}
    if [len(ties[s]) for s in ties] != [8, 8, 4, 2, 1]:
        raise ValueError('Cannot draw a bracket with incorrect knockout tie counts')

    # Play-off winners feed the R16 ties in a fixed bracket: reorder visually
    # by their *actual* next-round match rather than by play-off group number.
    p_by_winner = {tie['winner']: tie for tie in ties['playoff']}
    playoff_order = []
    for r16_tie in ties['r16']:
        names = _tie_participants(r16_tie)
        matches = [p_by_winner[name] for name in names if name in p_by_winner]
        if len(matches) != 1:
            raise ValueError(f'Cannot map play-off winner into R16 tie: {names}')
        playoff_order.append(matches[0])
    ties['playoff'] = playoff_order

    y16 = np.linspace(2.0, 10.55, 8)
    yq = np.array([np.mean(y16[i:i + 2]) for i in range(0, 8, 2)])
    ys = np.array([np.mean(yq[i:i + 2]) for i in range(0, 4, 2)])
    yf = np.array([np.mean(ys)])
    yy = {'playoff': y16, 'r16': y16, 'quarter_final': yq,
          'semi_final': ys, 'final': yf}
    xx = {'playoff': 0.55, 'r16': 4.10, 'quarter_final': 7.65,
          'semi_final': 11.20, 'final': 14.75}
    ww = 2.83
    xchamp = 18.30

    fig, ax = plt.subplots(figsize=figsize, facecolor='#f6f8fc')
    ax.set_facecolor('#f6f8fc')
    ax.set_xlim(0, 21.8)
    ax.set_ylim(12.05, 0)
    ax.axis('off')

    ink = '#14263d'
    muted = '#64748b'
    line = '#a7b7c9'
    green = '#08795c'

    ax.text(0.60, 0.48, 'UEFA CHAMPIONS LEAGUE  /  2026–27',
            fontsize=17, color=ink, weight='bold', va='center')
    ax.text(0.61, 0.89,
            f'Conditional representative knockout path  ·  {method.replace("_", " ").title()}'
            f'  ·  {selection["n_champion_paths"]:,}/{selection["n_simulations"]:,} champion paths',
            fontsize=10, color=muted, va='center')

    headers = [('playoff', 'KNOCKOUT PLAY-OFF'), ('r16', 'ROUND OF 16'),
               ('quarter_final', 'QUARTER-FINAL'), ('semi_final', 'SEMI-FINAL'),
               ('final', 'FINAL')]
    for stage, label in headers:
        ax.text(xx[stage] + 0.03, 1.32, label, fontsize=9, color=muted,
                weight='bold', va='center')
    ax.text(xchamp + 0.04, 1.32, 'PREDICTED CHAMPION', fontsize=9,
            color=muted, weight='bold', va='center')

    # Connector lines behind all cards. Tournament dependencies are real:
    # 8 R16 -> 4 quarterfinals -> 2 semifinals -> final -> champion.
    for i in range(8):
        ax.plot([xx['playoff'] + ww, xx['r16']], [y16[i], y16[i]],
                color=line, lw=1.1, zorder=1)
    for prev, nxt in [('r16', 'quarter_final'),
                      ('quarter_final', 'semi_final'), ('semi_final', 'final')]:
        src_ys, dst_ys = yy[prev], yy[nxt]
        midx = (xx[prev] + ww + xx[nxt]) / 2
        for j in range(len(dst_ys)):
            child1, child2 = src_ys[j * 2:j * 2 + 2]
            ax.plot([xx[prev] + ww, midx], [child1, child1],
                    color=line, lw=1.1, zorder=1)
            ax.plot([xx[prev] + ww, midx], [child2, child2],
                    color=line, lw=1.1, zorder=1)
            ax.plot([midx, midx], [child1, child2],
                    color=line, lw=1.1, zorder=1)
            ax.plot([midx, xx[nxt]], [dst_ys[j], dst_ys[j]],
                    color=line, lw=1.1, zorder=1)
    ax.plot([xx['final'] + ww, xchamp], [yf[0], yf[0]],
            color=line, lw=1.3, zorder=1)

    records = {(r.stage, int(r.slot)): r for r in table.itertuples(index=False)}

    def short(name, max_chars=23):
        return name if len(name) <= max_chars else name[:max_chars - 1] + '…'

    card_height = 0.96
    for stage in ('playoff', 'r16', 'quarter_final', 'semi_final', 'final'):
        for index, tie in enumerate(ties[stage]):
            rec = records[(stage, int(tie['slot']))]
            cx, cy = xx[stage], float(yy[stage][index])
            card = FancyBboxPatch(
                (cx, cy - card_height / 2), ww, card_height,
                boxstyle='round,pad=0.035,rounding_size=0.12',
                facecolor='white', edgecolor='#dce5ef', linewidth=0.9,
                zorder=3)
            ax.add_patch(card)
            # Highlight winning team with a small left indicator.
            ax.plot([cx + 0.09, cx + 0.09], [cy - 0.32, cy + 0.16],
                    color=green, lw=3.0, solid_capstyle='round', zorder=4)
            for line_idx, name in enumerate((rec.team_1, rec.team_2)):
                win = (name == rec.winner)
                ty = cy - 0.245 + 0.285 * line_idx
                ax.text(cx + 0.19, ty, short(name),
                        fontsize=8.4, weight='bold' if win else 'normal',
                        color=green if win else ink, va='center', zorder=5)
                if win:
                    ax.text(cx + ww - 0.15, ty, f'{rec.winner_probability:.0%}',
                            fontsize=8, color=green, weight='bold',
                            ha='right', va='center', zorder=5)
            score_text = rec.reference_score.replace(' · ET/PEN', ' · TB')
            ax.text(cx + 0.19, cy + 0.315, 'REF  ' + score_text,
                    fontsize=7.0, color=muted, va='center', zorder=5)

    champion = selection['champion']
    chance = None
    if championship_odds is not None:
        odds = championship_odds
        if isinstance(championship_odds, dict):
            odds = championship_odds['odds']
        row = odds.loc[odds['team'].eq(champion)]
        if not row.empty:
            chance = float(row.iloc[0]['champion'])

    champ_height = 1.28
    champ_card = FancyBboxPatch(
        (xchamp, yf[0] - champ_height / 2), 3.05, champ_height,
        boxstyle='round,pad=0.045,rounding_size=0.17',
        facecolor=ink, edgecolor=ink, linewidth=1, zorder=4)
    ax.add_patch(champ_card)
    ax.text(xchamp + 0.21, yf[0] - 0.32, 'WINNER',
            color='#b8dfcd', fontsize=8, weight='bold', zorder=5)
    ax.text(xchamp + 0.21, yf[0] + 0.02, short(champion, 21),
            color='white', fontsize=12, weight='bold', zorder=5)
    ax.text(xchamp + 0.21, yf[0] + 0.38,
            ('Title probability  ' + f'{chance:.1%}') if chance is not None
            else 'Conditional path selection',
            color='#d5e1ed', fontsize=8, zorder=5)

    foot = ('Percentage = P(highlighted team advances | this match-up); '
            'two-leg REF = most likely regulation-time aggregate conditional on winner. '
            'TB = extra time / penalties may be required.')
    ax.text(0.60, 11.33, foot, fontsize=7.8, color=muted)
    ax.text(0.60, 11.66,
            'Illustrative bracket selected from simulated paths conditional on the highest-probability champion; '
            'not the single most probable exact tournament outcome.',
            fontsize=7.8, color=muted)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return fig, table
