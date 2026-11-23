"""Team feature bank: one row per (season, club_id).

GK / DEF / MID / ATT quotas are soft except exactly one genuine GK is mandatory.
All within-position slots are sorted by Overall decreasing. Distinct IDs always.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd

# All versions have these numeric fields, although individual cells can be NA.
PLAYER_NUM = [
    'overall','potential','age','height_cm','weight_kg','weak_foot','skill_moves',
    'movement_acceleration','movement_sprint_speed','movement_agility','movement_reactions',
    'power_stamina','power_strength','power_shot_power','power_jumping',
    'attacking_finishing','attacking_short_passing','skill_dribbling','skill_ball_control',
    'skill_long_passing','mentality_vision','mentality_composure',
    'mentality_interceptions','defending_marking_awareness','defending_standing_tackle',
    'goalkeeping_diving','goalkeeping_handling','goalkeeping_kicking',
    'goalkeeping_positioning','goalkeeping_reflexes',
]
CLUB_NUM = ['overall','attack','midfield','defence','domestic_prestige','international_prestige']
ROLES = [('gk',1),('def',4),('mid',3),('att',3)]
# GK 1 + 4 DEF + 3 MID + 3 ATT, without hardcoding a 4-3-3 formation.
POS_ROLE = {
 'GK':'gk',
 'CB':'def','LB':'def','RB':'def','LWB':'def','RWB':'def',
 'CDM':'mid','CM':'mid','CAM':'mid','LM':'mid','RM':'mid',
 'ST':'att','CF':'att','LW':'att','RW':'att','LF':'att','RF':'att',
}

def positions(s):
    if pd.isna(s): return []
    return re.findall(r'\b(?:GK|CB|LB|RB|LWB|RWB|CDM|CM|CAM|LM|RM|ST|CF|LW|RW|LF|RF)\b', str(s).upper())

def get_role(row):
    ps=positions(row.player_positions)
    if not ps and isinstance(row.club_position,str):
        ps=positions(row.club_position)
    return POS_ROLE.get(ps[0], 'unknown') if ps else 'unknown'

def select_xi(squad):
    """Return (11 records, slot provenance) with stable GK/DEF/MID/ATT slots.

    Fill missing positions IN THEIR OWN GROUP, not at the end of the XI.
    This preserves def1..4, mid1..3, att1..3 semantics across teams.
    """
    sq = squad.copy().sort_values(['overall', 'player_id'], ascending=[False, True], na_position='last')
    if '__role' not in sq.columns:
        sq['__role'] = sq.apply(get_role, axis=1)
    if len(sq) < 11 or not (sq['__role'] == 'gk').any():
        return None
    by_role = {}
    used = set()
    for role, count in ROLES:
        candidates = sq[(sq['__role'] == role) & (~sq.player_id.isin(used))].head(count)
        by_role[role] = candidates.to_dict('records')
        used.update(candidates.player_id.tolist())
    # Only non-GK gaps can be filled; every selected GK is a genuine goalkeeper.
    remaining = sq[~sq.player_id.isin(used)].to_dict('records')
    idx = 0
    selected, provenance = [], []
    for role, count in ROLES:
        chosen = by_role[role]
        for player in chosen:
            selected.append(player)
            provenance.append(role)
        gap = count - len(chosen)
        if role == 'gk' and gap:
            return None
        for _ in range(gap):
            if idx >= len(remaining):
                return None
            selected.append(remaining[idx])
            provenance.append('fill')
            idx += 1
    if len(selected) != 11 or len({r['player_id'] for r in selected}) != 11:
        return None
    assert selected[0]['__role'] == 'gk'
    return selected, provenance

def mean(vals):
    z=pd.to_numeric(pd.Series(list(vals),dtype='object'),errors='coerce')
    return float(z.mean()) if z.notna().any() else np.nan

def numeric_summaries(players, prefix):
    out={}
    for col in ['overall','potential','age','height_cm','weight_kg','movement_sprint_speed',
                'attacking_finishing','attacking_short_passing','defending_standing_tackle',
                'goalkeeping_reflexes']:
        vals=[p.get(col,np.nan) for p in players]
        out[f'{prefix}_{col}_mean']=mean(vals)
    ovs=pd.to_numeric(pd.Series([p.get('overall') for p in players]),errors='coerce')
    out[f'{prefix}_overall_max']=float(ovs.max()) if ovs.notna().any() else np.nan
    out[f'{prefix}_overall_std']=float(ovs.std(ddof=0)) if ovs.notna().any() else np.nan
    return out

def build_team_features(players,clubs,club_features):
    """Returns (full team bank, quality flags).

    Top11 slot names: gk1, def1..def4, mid1..mid3, att1..att3.
    When role quotas cannot be filled, slots are populated by best remaining
    outfield players and flagged; each slot preserves deterministic order.
    """
    required={'season','club_id','player_id','overall','player_positions'}
    missing=required-set(players.columns)
    if missing:raise ValueError(f'players missing required columns: {missing}')
    missing=required.intersection({'season','club_id'})-set(clubs.columns)
    if missing:raise ValueError(f'clubs missing columns: {missing}')
    p=players.copy();c=clubs.copy(); cf=club_features.copy()
    for d in [p,c,cf]:
        d['club_id']=pd.to_numeric(d.club_id,errors='coerce').astype('Int64')
        d['season']=d.season.astype(str)
    p['overall']=pd.to_numeric(p['overall'],errors='coerce')
    p=p.dropna(subset=['club_id','player_id','overall'])
    # One person per club/season; source should already be deduplicated.
    if p.duplicated(['season','club_id','player_id']).any():
        raise ValueError('Duplicated season,club,player IDs in player data')
    for col in PLAYER_NUM:
        if col not in p:p[col]=np.nan
        p[col]=pd.to_numeric(p[col],errors='coerce')
    # Extract primary player role ONCE, instead of re-parsing all player rows
    # for every club and once again for the GK availability flag.
    first_pos=p['player_positions'].fillna('').astype(str).str.upper().str.extract(
       r'\b(GK|CB|LB|RB|LWB|RWB|CDM|CM|CAM|LM|RM|ST|CF|LW|RW|LF|RF)\b', expand=False)
    second_pos=p['club_position'].fillna('').astype(str).str.upper().str.extract(
       r'\b(GK|CB|LB|RB|LWB|RWB|CDM|CM|CAM|LM|RM|ST|CF|LW|RW|LF|RF)\b', expand=False)
    p['__role']=first_pos.fillna(second_pos).map(POS_ROLE).fillna('unknown')
    for col in CLUB_NUM:
        if col not in c:c[col]=np.nan
        c[col]=pd.to_numeric(c[col],errors='coerce')
    if c.duplicated(['season','club_id']).any():raise ValueError('Duplicate Club season keys')
    if cf.duplicated(['season','club_id']).any():raise ValueError('Duplicate club_features season keys')
    by_squad={k:g for k,g in p.groupby(['season','club_id'],sort=False)}
    result=[];status=[]
    slot_names=['gk1']+[f'def{i}' for i in range(1,5)]+[f'mid{i}' for i in range(1,4)]+[f'att{i}' for i in range(1,4)]
    for row in c.itertuples(index=False):
        season=str(row.season);club_id=int(row.club_id)
        sq=by_squad.get((season,club_id))
        chosen=select_xi(sq) if sq is not None else None
        flag={'season':season,'club_id':club_id,'club_name':row.club_name,
              'roster_size':len(sq) if sq is not None else 0,
              'has_gk':bool(sq is not None and (sq['__role']=='gk').any()),
              'has_xi':chosen is not None}
        status.append(flag)
        if chosen is None: continue
        picked,roles=chosen
        used={r['player_id'] for r in picked}
        bench=sq[~sq.player_id.isin(used)].to_dict('records')
        obj={'season':season,'club_id':club_id,'club_name':row.club_name,
             'xi_fill_count':roles.count('fill'),'squad_count':len(sq),
             'bench_count':len(bench),'xi_left_foot_count':sum(str(r.get('preferred_foot','')).lower()=='left' for r in picked)}
        for col in CLUB_NUM:obj[f'club_{col}']=getattr(row,col)
        # Recalculate top11 / bench aggregates from the actual selected 11,
        # rather than reusing the old naive Overall-top11 features.
        obj.update(numeric_summaries(picked,'xi'))
        obj.update(numeric_summaries(bench,'bench'))
        for cutoff in [75,80,85]:
            obj[f'bench_overall_ge_{cutoff}']=sum(float(r.get('overall',0))>=cutoff for r in bench)
        for role in ['gk','def','mid','att']:
            sub=[r for r in bench if r.get('__role',get_role(pd.Series(r)))==role]
            obj[f'bench_{role}_count']=len(sub)
            obj[f'bench_{role}_best_overall']=max((r['overall'] for r in sub),default=np.nan)
        for slot,r in zip(slot_names,picked):
            for col in PLAYER_NUM: obj[f'{slot}_{col}']=r.get(col,np.nan)
            obj[f'{slot}_preferred_foot_left']=int(str(r.get('preferred_foot','')).lower()=='left')
            obj[f'{slot}_role_mismatch']=int(r['__role']!=('gk' if slot.startswith('gk') else 'def' if slot.startswith('def') else 'mid' if slot.startswith('mid') else 'att'))
        # Historical precomputed features are intentionally NOT used here;
        # their top11 was position-agnostic and could disagree with selected XI.
        result.append(obj)
    bank=pd.DataFrame(result)
    flags=pd.DataFrame(status)
    if bank.duplicated(['season','club_id']).any():raise ValueError('Duplicate team bank keys')
    return bank,flags
