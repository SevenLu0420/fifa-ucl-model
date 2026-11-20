# python run_eda.py --data output_final --coverage coverage_audit/ucl_teams_coverage.csv --out eda_output
from __future__ import annotations
import argparse
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEASONS = [f'{y}-{str(y+1)[-2:]}' for y in range(2014,2027)]
CORE_PLAYER = ['age','height_cm','weight_kg','overall','potential','pace','shooting','passing','dribbling','defending','physic','value_eur','wage_eur','release_clause_eur']
CORE_CLUB = ['overall','attack','midfield','defence','domestic_prestige','international_prestige','club_worth_eur']
TECH = ['attacking_finishing','attacking_short_passing','skill_ball_control','movement_acceleration','movement_sprint_speed','power_stamina','mentality_vision','defending_marking_awareness','goalkeeping_reflexes']


def read_csv(path):
    if not path.exists(): raise FileNotFoundError(f'Required data missing: {path}')
    return pd.read_csv(path,low_memory=False)


def numeric(s): return pd.to_numeric(s,errors='coerce')


def write(df,path):
    df.to_csv(path,index=False,encoding='utf-8-sig')


def chart(fig,path):
    fig.tight_layout();fig.savefig(path,dpi=170,bbox_inches='tight');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='output_final',help='folder containing 4 normalized CSVs')
    p.add_argument('--coverage',default='coverage/ucl_teams_coverage.csv',help='UCL team coverage CSV, optional')
    p.add_argument('--out',default='eda_output')
    a=p.parse_args();data=Path(a.data);out=Path(a.out);report=out/'tables';plots=out/'figures';report.mkdir(parents=True,exist_ok=True);plots.mkdir(parents=True,exist_ok=True)
    players=read_csv(data/'players_2015_2027.csv');clubs=read_csv(data/'clubs_2015_2027.csv');features=read_csv(data/'club_features_2015_2027.csv');coaches=read_csv(data/'coaches_2015_2027.csv')
    coverage=read_csv(Path(a.coverage)) if Path(a.coverage).exists() else None
    for df in (players,clubs,features,coaches):
        df['season']=df['season'].astype(str)
    if coverage is not None: coverage['season']=coverage['season'].astype(str)
    # Validate basic integrity, but preserve outputs if the input contains issues.
    qc=[]
    for label,df,key in [('players',players,['season','player_id','club_id']),('clubs',clubs,['season','club_id']),('features',features,['season','club_id']),('coaches',coaches,['season','club_id'])]:
        qc.append({'table':label,'rows':len(df),'columns':len(df.columns),'duplicate_keys':int(df.duplicated(key).sum()),'seasons':df.season.nunique()})
    valid_keys=clubs[['season','club_id']].drop_duplicates().assign(_valid=1)
    invalid=players[['season','club_id']].merge(valid_keys,on=['season','club_id'],how='left')['_valid'].isna().sum()
    qc.append({'table':'player_club_fk','rows':len(players),'columns':np.nan,'duplicate_keys':int(invalid),'seasons':players.season.nunique()})
    write(pd.DataFrame(qc),report/'00_integrity.csv')

    # Team-season universe: all clubs vs matched UCL clubs.
    if coverage is not None and {'season','club_id','status'}.issubset(coverage.columns):
        match=coverage.loc[coverage['status'].astype(str).str.startswith('MATCHED'),['season','club_id']].dropna().copy()
        match['club_id']=numeric(match['club_id']);match=match.drop_duplicates()
        if match.duplicated(['season','club_id']).any(): warnings.warn('Duplicated UCL club mappings found')
    else:
        match=pd.DataFrame(columns=['season','club_id']);warnings.warn('No UCL mapping: UCL-only analyses skipped')
    clubs['club_id']=numeric(clubs['club_id']);features['club_id']=numeric(features['club_id']);players['club_id']=numeric(players['club_id'])
    clubs=clubs.merge(match.assign(is_ucl=1),on=['season','club_id'],how='left');clubs['is_ucl']=clubs.is_ucl.fillna(0).astype(int)
    features=features.merge(match.assign(is_ucl=1),on=['season','club_id'],how='left');features['is_ucl']=features.is_ucl.fillna(0).astype(int)
    players=players.merge(match.assign(is_ucl=1),on=['season','club_id'],how='left');players['is_ucl']=players.is_ucl.fillna(0).astype(int)
    coaches=coaches.merge(match.assign(is_ucl=1),on=['season','club_id'],how='left');coaches['is_ucl']=coaches.is_ucl.fillna(0).astype(int)
    # Basic season sizes and squad stats.
    season=pd.DataFrame(index=SEASONS)
    season.index.name='season'
    season['players_all']=players.groupby('season').size()
    season['clubs_all']=clubs.groupby('season').size()
    season['players_ucl']=players.loc[players.is_ucl.eq(1)].groupby('season').size()
    season['clubs_ucl']=clubs.loc[clubs.is_ucl.eq(1)].groupby('season').size()
    season['players_unique_all']=players.groupby('season').player_id.nunique()
    season['coach_named_all']=coaches.groupby('season')['name'].apply(lambda s:s.notna().sum())
    season['coach_name_rate_all']=coaches.groupby('season')['name'].apply(lambda s:s.notna().mean())
    write(season.reset_index(),report/'01_season_summary.csv')
    # UCL roster coverage.
    if coverage is not None:
        cov=coverage.copy();cov['matched']=cov['status'].astype(str).str.startswith('MATCHED')
        t=cov.groupby('season').agg(uefa_teams=('matched','size'),matched=('matched','sum')).reindex(SEASONS)
        t['unmatched']=t.uefa_teams-t.matched;t['match_rate']=t.matched/t.uefa_teams
        write(t.reset_index(),report/'02_ucl_coverage.csv')
        fig,ax=plt.subplots(figsize=(11,4.5));ax.bar(t.index,t.matched,label='Matched');ax.bar(t.index,t.unmatched,bottom=t.matched,label='Unmatched')
        ax.set_title('UCL club coverage by season');ax.set_ylabel('Teams');ax.legend();ax.tick_params(axis='x',rotation=50);chart(fig,plots/'01_ucl_club_coverage.png')
        write(cov.loc[~cov.matched],report/'03_unmatched_ucl_clubs.csv')
    # Coverage heatmaps for useful numeric feature groups.
    rows=[]
    for table_name,df,cols in [('player',players,CORE_PLAYER+TECH),('club',clubs,CORE_CLUB),('feature',features,[c for c in features.columns if c not in ('season','club_id','club_name','snapshot_date','is_ucl')])]:
        for scope in ['all','ucl']:
            dd=df if scope=='all' else df.loc[df.is_ucl.eq(1)]
            for col in cols:
                if col not in dd.columns: continue
                by=dd.groupby('season')[col].apply(lambda s:s.notna().mean()).reindex(SEASONS)
                rows.extend({'table':table_name,'scope':scope,'season':k,'feature':col,'non_null_rate':v} for k,v in by.items())
    rate=pd.DataFrame(rows)
    write(rate,report/'04_feature_availability.csv')
    for tab,cols in [('player',CORE_PLAYER+TECH),('club',CORE_CLUB)]:
        q=rate.loc[(rate.table==tab)&(rate.scope=='ucl')].pivot(index='feature',columns='season',values='non_null_rate')
        if len(q)==0:continue
        q=q.reindex([x for x in cols if x in q.index]);fig,ax=plt.subplots(figsize=(12,max(5,len(q)*.31)))
        im=ax.imshow(q.fillna(0).to_numpy(),aspect='auto',vmin=0,vmax=1,cmap='viridis')
        ax.set_yticks(range(len(q)),labels=q.index,fontsize=8);ax.set_xticks(range(len(q.columns)),labels=q.columns,rotation=55,ha='right',fontsize=8)
        ax.set_title(f'{tab.title()} availability - matched UCL clubs');fig.colorbar(im,ax=ax,label='Non-null fraction',shrink=.7)
        chart(fig,plots/f'02_{tab}_availability.png')
    # Year-by-year distribution of club scores (ALL and UCL, avoid temporal line of global pool).
    ratings=[]
    for scope in ['all','ucl']:
        d=clubs if scope=='all' else clubs[clubs.is_ucl.eq(1)]
        for col in ['overall','attack','midfield','defence']:
            if col not in d.columns:continue
            g=d.groupby('season')[col].agg(['count','mean','median','std','min','max']).reindex(SEASONS).reset_index();g['scope']=scope;g['attribute']=col;ratings.append(g)
    rating=pd.concat(ratings,ignore_index=True)
    write(rating,report/'05_club_rating_distributions.csv')
    fig,ax=plt.subplots(figsize=(11,4.5))
    for scope in ['all','ucl']:
        g=rating[(rating.scope==scope)&(rating.attribute=='overall')];ax.plot(g.season,g['median'],marker='o',label=f'{scope} median')
    ax.set_title('Club Overall median by season (descriptive only)');ax.set_ylabel('Game rating');ax.tick_params(axis='x',rotation=55);ax.legend();chart(fig,plots/'03_club_overall_trend.png')
    # Difference between native club rating and player-aggregated best 11.
    top=features.copy();top['top11_minus_club_overall']=numeric(top.overall_top11_mean)-numeric(top.club_overall)
    top['top11_minus_top16']=numeric(top.overall_top11_mean)-numeric(top.overall_top16_mean)
    fields=['season','club_id','club_name','is_ucl','squad_count','club_overall','overall_top11_mean','overall_top16_mean','top11_minus_club_overall','top11_minus_top16','squad_depth_75','squad_depth_80','squad_depth_85']
    write(top[fields],report/'06_team_strength_and_depth.csv')
    for scope in ['all','ucl']:
        s=top if scope=='all' else top[top.is_ucl.eq(1)]
        grouped=s.groupby('season')[['squad_count','overall_top11_mean','overall_top16_mean','top11_minus_club_overall','top11_minus_top16']].median().reindex(SEASONS).reset_index();grouped.insert(1,'scope',scope)
        write(grouped,report/f'07_{scope}_team_strength_summary.csv')
    fig,ax=plt.subplots(figsize=(11,4.5));s=top[top.is_ucl.eq(1)]
    ax.scatter(numeric(s.club_overall),numeric(s.overall_top11_mean),s=12,alpha=.35)
    ax.set_xlabel('Native club Overall');ax.set_ylabel('Top11 mean Overall');ax.set_title('Native game rating vs player Top11 (matched UCL teams)');chart(fig,plots/'04_club_vs_top11.png')
    # Player age, overall, potential, and technical ratings (not a causal comparison).
    sel=players[players.is_ucl.eq(1)].copy();sel['growth_gap']=numeric(sel.potential)-numeric(sel.overall)
    annual=sel.groupby('season').agg(n=('player_id','size'),overall_median=('overall','median'),overall_mean=('overall','mean'),potential_mean=('potential','mean'),age_median=('age','median'),growth_gap_mean=('growth_gap','mean')).reindex(SEASONS).reset_index()
    write(annual,report/'08_ucl_player_season_summary.csv')
    fig,ax=plt.subplots(figsize=(11,4.5))
    ax.plot(annual.season,annual.overall_mean,marker='o',label='Overall mean')
    ax.plot(annual.season,annual.potential_mean,marker='o',label='Potential mean')
    ax.set_title('Player ratings in UCL matched-club squads');ax.set_ylabel('Game rating');ax.tick_params(axis='x',rotation=55);ax.legend();chart(fig,plots/'05_player_overall_potential.png')
    # Position group approximate using explicit positional strings; avoid SUB/RES as a 'role'.
    def position_group(v):
        if pd.isna(v):return 'Unknown'
        s=str(v).upper().replace(';',',').replace('/ ', ',')
        tokens=[x.strip() for x in s.replace('/',',').split(',')]
        if 'GK' in tokens:return 'GK'
        if any(t in {'CB','LB','RB','LWB','RWB','SW','LCB','RCB'} for t in tokens):return 'DEF'
        if any(t in {'CDM','CM','CAM','LM','RM','LCM','RCM','LDM','RDM','LAM','RAM'} for t in tokens):return 'MID'
        if any(t in {'ST','CF','LW','RW','LF','RF','LS','RS'} for t in tokens):return 'ATT'
        return 'Unknown'
    sel['pos_group']=sel.player_positions.apply(position_group)
    pos=sel.groupby(['season','pos_group']).size().unstack(fill_value=0).reindex(SEASONS).fillna(0)
    write(pos.reset_index(),report/'09_ucl_position_counts.csv')
    position_order=[x for x in ['GK','DEF','MID','ATT','Unknown'] if x in pos.columns]
    fig,ax=plt.subplots(figsize=(11,4.5));ax.stackplot(pos.index,[pos[c].to_numpy() for c in position_order],labels=position_order)
    ax.set_title('UCL player position groups (approximate)');ax.set_ylabel('Player–club records');ax.tick_params(axis='x',rotation=55);ax.legend(loc='upper left',ncol=5);chart(fig,plots/'06_position_structure.png')
    # Inter-year same-player deltas: descriptive ONLY, may depend on snapshot differences.
    unique=players.sort_values('season').drop_duplicates(['season','player_id'])[['season','player_id','overall','potential','age']].copy()
    unique['edition']=unique.season.str[:4].astype(int)
    before=unique.rename(columns={'season':'prev_season','overall':'prev_overall','potential':'prev_potential','age':'prev_age'})
    before['edition']=before.edition+1
    deltas=unique.merge(before[['player_id','edition','prev_season','prev_overall','prev_potential','prev_age']],on=['player_id','edition'],how='inner')
    deltas['delta_overall']=numeric(deltas.overall)-numeric(deltas.prev_overall)
    deltas['delta_potential']=numeric(deltas.potential)-numeric(deltas.prev_potential)
    write(deltas[['season','prev_season','player_id','overall','prev_overall','delta_overall','potential','prev_potential','delta_potential']],report/'10_player_year_on_year_changes.csv')
    transition=deltas.groupby('season').delta_overall.agg(['count','mean','median','std']).reset_index()
    write(transition,report/'11_player_rating_change_summary.csv')
    # Coach quality audit: flag that a coach's record may reflect future appointment.
    coach=coaches.copy();coach['name_present']=coach.name.notna()
    coach_rate=coach.groupby(['season','is_ucl']).agg(coach_rows=('coach_id','size'),coach_id_non_null=('coach_id',lambda s:s.notna().sum()),name_non_null=('name_present','sum')).reset_index()
    coach_rate['name_rate']=coach_rate.name_non_null/coach_rate.coach_rows
    write(coach_rate,report/'12_coach_coverage.csv')
    # Finance and tactics availability only (do not convert, nor assert comparability).
    tactical=[c for c in clubs if c.startswith('tactic_') and not c.endswith('_version')]
    records=[]
    for scope,dd in [('all',clubs),('ucl',clubs[clubs.is_ucl.eq(1)])]:
        for col in tactical:
            for season_name,g in dd.groupby('season'):
                records.append({'scope':scope,'season':season_name,'feature':col,'non_null_rate':g[col].notna().mean(),'distinct_non_null':g[col].nunique(dropna=True)})
    write(pd.DataFrame(records),report/'13_tactics_availability.csv')
    fig,ax=plt.subplots(figsize=(11,4.5))
    tactic_av=pd.DataFrame(records);t=tactic_av[tactic_av.scope=='ucl'].groupby('season').agg(num_tactic_columns_available=('non_null_rate',lambda x:(x>.1).sum())).reindex(SEASONS)
    ax.bar(t.index,t.num_tactic_columns_available)
    ax.set_title('Tactic columns with >10% coverage in matched UCL clubs');ax.set_ylabel('Columns');ax.tick_params(axis='x',rotation=55);chart(fig,plots/'07_tactic_columns_by_season.png')
    # Cross-sectional correlation in UCL club features; diagnostics not model selection.
    cols=[x for x in ['club_overall','club_attack','club_midfield','club_defence','overall_top11_mean','overall_top16_mean','squad_count','squad_depth_80','age_top11_mean'] if x in features]
    correlation=features.loc[features.is_ucl.eq(1),cols].apply(numeric).corr()
    correlation.to_csv(report/'14_ucl_feature_correlations.csv',encoding='utf-8-sig')
    fig,ax=plt.subplots(figsize=(9,7));im=ax.imshow(correlation.to_numpy(),vmin=-1,vmax=1,cmap='coolwarm')
    ax.set_xticks(range(len(cols)),labels=cols,rotation=65,ha='right',fontsize=8);ax.set_yticks(range(len(cols)),labels=cols,fontsize=8);fig.colorbar(im,ax=ax)
    ax.set_title('UCL matched club feature correlations');chart(fig,plots/'08_feature_correlations.png')
    # Summary and important caveats.
    stats={'data_rows':{'players':int(len(players)),'clubs':int(len(clubs)),'coaches':int(len(coaches)),'club_features':int(len(features))},
           'seasons':int(clubs.season.nunique()), 'matched_ucl_team_seasons':int(match.shape[0]),
           'age_missing':int(players.age.isna().sum()), 'player_duplicate_keys':int(players.duplicated(['season','player_id','club_id']).sum()),
           'coach_warning':'Historical coach names may not reflect historical appointments; verify before use.',
           'scope_note':'UCL club analysis covers matched clubs only, not missing club participants.',
           'top11_warning':'Precomputed Top11 is based on overall ranking, not position-constrained lineup.',
           'season_warning':'Descriptive changes across editions may reflect game balance changes, snapshot differences, and squad composition; no causal interpretation.',
           'no_match_data':'No UCL match scores or labels used.'}
    (out/'summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(stats,indent=2,ensure_ascii=False));print(f'Finished: {report} and {plots}')

if __name__=='__main__':main()
