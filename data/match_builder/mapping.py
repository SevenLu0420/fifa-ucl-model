"""Conservative match-team -> season-specific EA club IDs.

Never automatically accept fuzzy matches. All manually audited aliases below resolve
only when the resulting club ID exists in the selected season's Club table.
"""
from __future__ import annotations
import re
import unicodedata
from collections import defaultdict
import pandas as pd

# Match-data names from the supplied UEFA workbook; EA IDs verified against
# historical name mapping or same franchise identifiers (not fuzzy guesses).
ALIASES = {
    'SSC Napoli': 48,       # Official SSC Napoli = Napoli; verified FC26 club_id 48
    'Bayer Leverkusen': 32,
    'PSV Eindhoven': 247,
    'Monaco': 69,
    'Brest': 378,
    'Inter Milan': 131682,   # EA FC25+: Lombardia FC; never use historical Inter ID=44 here
    'Milan': 131681,         # EA FC25+: Milano FC; older seasons use 47 if needed
    'Bayern Munich': 21,
    'Copenhagen': 819,
    'Lyon': 66,
    'CSKA Moscow': 315,
    'Olympiacos': 280,
    'Basel': 896,
    'Marseille': 219,
    'Gent': 674,
    'Legia Warsaw': 1871,
    'Anderlecht': 229,
    'Spartak Moscow': 100767,
    'Genk': 673,
    'Slavia Prague': 266,
    'Braga': 1896,
    'Sparta Prague': 267,
}

# Exact, manually identified UEFA-name equivalents. These do not imply a
# confirmed club link by themselves: the target ID must be present in a
# season-specific verified coverage/known-map AND that game's Club table.
NAME_EQUIVALENTS = {
    'FC Internazionale Milano':'Inter',
    'FC Bayern München':'Bayern München',
    'Club Atlético de Madrid':'Atlético de Madrid',
    'Bayer 04 Leverkusen':'Bayer Leverkusen',
    'Club Brugge KV':'Club Brugge',
    'Atalanta BC':'Atalanta',
    'AS Monaco FC':'Monaco',
    'Galatasaray SK':'Galatasaray',
    'Juventus FC':'Juventus',
    'Manchester City FC':'Manchester City',
    'Newcastle United FC':'Newcastle United',
    'Olympique de Marseille':'Marseille',
    'PAE Olympiakos SFP':'Olympiacos',
    'Olympiakos Piraeus':'Olympiacos',
    'Qarabağ Ağdam FK':'Qarabağ',
    'Royale Union Saint-Gilloise':'Union Saint-Gilloise',
    'SK Slavia Praha':'Slavia Praha',
    'Sport Lisboa e Benfica':'Benfica',
    'Sporting Clube de Portugal':'Sporting CP',
    'Tottenham Hotspur FC':'Tottenham Hotspur',
    'Villarreal CF':'Villarreal',
    'FK Bodø/Glimt':'Bodø/Glimt',
    'FC København':'FC Copenhagen',
    'Leipzig':'RB Leipzig',
    'Inter Milan':'Inter',
    'Milan':'AC Milan',
    'PSV':'PSV Eindhoven',
    'AEK Athens':'AEK Athens',
    'Stuttgart':'Stuttgart',
    'Lille':'Lille',
    'Viking':'Viking',
    'Arsenal FC':'Arsenal',
    'Chelsea FC':'Chelsea',
    'Liverpool FC':'Liverpool',
    'Real Madrid CF':'Real Madrid',
    'Paris Saint-Germain FC':'Paris Saint-Germain',
    'Atlético Madrid':'Atlético de Madrid',
}

def norm(v):
    x = unicodedata.normalize('NFKD', str(v or '')).encode('ascii', 'ignore').decode().lower()
    x = re.sub(r'\b(fc|cf|afc|fk|sc|ac|cfc|club|football|futbol|futebol|soccer)\b', ' ', x)
    return re.sub(r'[^a-z0-9]', '', x)

def as_id(v):
    if pd.isna(v) or str(v).strip() == '':
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None

def make_mapping(matches, clubs, coverage, reviewed=None, known=None):
    """Returns mapping table + a textual review report for unresolved names.

    Resolution order: manual ID aliases; exact match; verified UEFA coverage
    (including alias equivalent); unique cross-season proven identity. No fuzz.
    """
    club_by_key = {(str(r.season), int(r.club_id)): r.club_name for r in clubs.itertuples()}
    by_season = defaultdict(list)
    by_name_all = defaultdict(set)
    for r in clubs.itertuples():
        by_season[str(r.season)].append((int(r.club_id), r.club_name))
        by_name_all[norm(r.club_name)].add(int(r.club_id))
    cov_by_season = defaultdict(list)
    for r in coverage.itertuples(index=False):
        rid = as_id(getattr(r, 'club_id', None))
        if rid is not None:
            cov_by_season[str(r.season)].append((norm(r.uefa_team), rid, str(r.uefa_team)))
    # Names successfully checked in an earlier audit can be reused, but IDs
    # must again exist in the matching FIFA season.
    reviewed_ids = {}
    known_ids = {}
    if known is not None and {'season','uefa_team','team_id','status'}.issubset(known):
        for r in known.itertuples(index=False):
            cid=as_id(r.team_id)
            if cid is not None and str(r.status) in {'exact_or_alias','alias_confirmed','MATCHED_SELECTED'}:
                key=(str(r.season),norm(r.uefa_team))
                if key in known_ids and known_ids[key]!=cid:
                    raise ValueError(f'Conflicting verified club ID for {key}')
                known_ids[key]=cid
    if reviewed is not None and {'season','match_team','club_id','mapping_status'}.issubset(reviewed):
        for r in reviewed.itertuples(index=False):
            if r.mapping_status == 'MATCHED' and as_id(r.club_id) is not None:
                reviewed_ids[(str(r.season), norm(r.match_team))] = as_id(r.club_id)
    rows=[]
    for season, name in sorted({(str(s), str(t)) for s,t in zip(matches.season, matches.home_team)} |
                               {(str(s), str(t)) for s,t in zip(matches.season, matches.away_team)}):
        nk = norm(name)
        alias_nk=norm(NAME_EQUIVALENTS.get(name,name))
        available = dict(by_season[season])
        chosen = None; method='unmatched'; candidates=[]
        # 1. Explicit alias; only if the club exists in this game's roster.
        forced=ALIASES.get(name)
        if forced in available:
            chosen=forced;method='audited_alias'
        # 2. Season-specific manual club map from repo reference.
        if chosen is None:
            verified=known_ids.get((season,alias_nk))
            if verified in available:
                chosen=verified;method='verified_2024_2027_map'
        # 3. Manually/audit-confirmed season-specific ID
        if chosen is None:
            old=reviewed_ids.get((season,nk))
            if old in available:
                chosen=old;method='previously_verified_audit'
        # 3. Unique exact normalized in game
        if chosen is None:
            hit={cid for cid, club_name in by_season[season] if norm(club_name) in {nk,alias_nk}}
            if len(hit)==1:
                chosen=next(iter(hit));method='exact_normalized'
        # 4. Unique verified UEFA coverage match in this season
        if chosen is None:
            aliases={nk,alias_nk}
            if name == 'Inter Milan': aliases.add(norm('Inter'))
            if name == 'Milan': aliases.add(norm('AC Milan'))
            hit={cid for u,cid,_ in cov_by_season[season] if u in aliases and cid in available}
            if len(hit)==1:
                chosen=next(iter(hit));method='verified_uefa_coverage'
        # 5. Cross-year game name match only when unique ID globally
        if chosen is None:
            hit={cid for key in {nk,alias_nk} for cid in by_name_all.get(key,set()) if cid in available}
            if len(hit)==1:
                chosen=next(iter(hit));method='cross_season_verified_id'
        if chosen is None:
            # suggestions are NEVER automatically mapped
            import difflib
            candidates=[f'{nm} [{cid}]' for cid,nm in sorted(by_season[season],
                  key=lambda x:difflib.SequenceMatcher(None,nk,norm(x[1])).ratio(),reverse=True)[:3]]
        rows.append(dict(season=season,match_team=name,club_id=chosen,club_name=available.get(chosen),
                         mapping_method=method, status='MATCHED' if chosen is not None else 'UNMATCHED',
                         suggestions=' | '.join(candidates)))
    df=pd.DataFrame(rows)
    if df.duplicated(['season','match_team']).any():
        raise ValueError('Duplicate mapping keys')
    return df
