"""FIFA15-FC27 postprocess from 4 standardized output CSVs.
No arbitrary median filling; immutable player attributes and DOB are carried across seasons if consistent.
Historical and FC25-27 tactics recovered from original game source files, with versioned raw codes.
"""
from __future__ import annotations
import argparse, io, json, re, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

TAC_COLS=['def_style','def_team_width','def_team_depth','def_defence_pressure','def_defence_aggression','def_defence_width','def_defence_defender_line','off_style','off_build_up_play','off_chance_creation','off_team_width','off_players_in_box','off_corners','off_free_kicks','build_up_play_speed','build_up_play_dribbling','build_up_play_passing','build_up_play_positioning','chance_creation_passing','chance_creation_crossing','chance_creation_shooting','chance_creation_positioning']
SEMANTIC={'def_style':'defensive_style','def_team_width':'defensive_width','def_team_depth':'defensive_depth','def_defence_pressure':'defensive_pressure','def_defence_aggression':'defensive_aggression','def_defence_width':'defensive_width_alt','def_defence_defender_line':'defender_line','off_style':'attacking_style','off_build_up_play':'build_up_play','off_chance_creation':'chance_creation','off_team_width':'attacking_width','off_players_in_box':'players_in_box','off_corners':'corners_commitment','off_free_kicks':'free_kicks_commitment','build_up_play_speed':'build_up_speed','build_up_play_dribbling':'build_up_dribbling','build_up_play_passing':'build_up_passing','build_up_play_positioning':'build_up_positioning','chance_creation_passing':'chance_passing','chance_creation_crossing':'chance_crossing','chance_creation_shooting':'chance_shooting','chance_creation_positioning':'chance_positioning'}
TEXT_TO_KEEP_PLAYER={'season','snapshot_date','name','player_positions','club_position','work_rate','player_traits','playstyles','age_fill_method','age_fill_reference','finance_source'}
TEXT_TO_DROP_PLAYER=['nationality_name','snapshot_source','snapshot_timing','rating_source','finance_snapshot_date','squad_role']
TEXT_TO_DROP_CLUB=['nationality_name','league_name','snapshot_source','snapshot_timing','manager_name','source_team_rating']

def load_csv(p):return pd.read_csv(p,low_memory=False,encoding='utf-8-sig')
def numeric(s): return pd.to_numeric(s,errors='coerce')
def season_year(s):return pd.to_numeric(s.astype(str).str[:4],errors='coerce')
def consistent_nonnull(x):
    y=x.dropna().unique()
    return y[0] if len(y)==1 else np.nan

def fill_player_age(p):
    p=p.copy();p['age']=numeric(p['age']);p['age_fill_method']=np.where(p.age.notna(),'original','missing');p['age_fill_reference']=''
    p['dob_parsed']=pd.to_datetime(p['dob'],errors='coerce') if 'dob' in p else pd.NaT
    p['ref_date']=pd.to_datetime(p['snapshot_date'],errors='coerce')
    # Date only used when dated snapshot is available; do not confuse future year with source snapshot.
    p['ref_date']=p.ref_date.fillna(pd.to_datetime(season_year(p.season).astype('Int64').astype(str)+'-09-01',errors='coerce'))
    # Same player's DOB can be reused from other observed years, when unique and coherent.
    p['candidate_dob']=pd.to_datetime(p.groupby('player_id')['dob_parsed'].transform(consistent_nonnull),errors='coerce')
    missing=p.age.isna(); known_dob=p.candidate_dob.notna() & p.ref_date.notna()
    exact=((p.ref_date.dt.year-p.candidate_dob.dt.year)-((p.ref_date.dt.month*100+p.ref_date.dt.day)<(p.candidate_dob.dt.month*100+p.candidate_dob.dt.day)).astype(int))
    valid_age=exact.between(15,55)
    mask=missing&known_dob&valid_age
    p.loc[mask,'age']=exact.loc[mask];p.loc[mask,'age_fill_method']='dob_cross_season_or_current';p.loc[mask,'age_fill_reference']='DOB'
    # For players with no DOB, same player in nearby seasons; exclude inconsistent observations.
    missing=p.age.isna()
    if missing.any():
        p['season_start']=season_year(p.season)
        age_known=p.loc[p.age.notna(),['player_id','season_start','age']].copy()
        age_known['birth_year_est']=age_known.season_start-age_known.age
        birth_stats=age_known.groupby('player_id').birth_year_est.agg(['median','min','max','count'])
        p=p.join(birth_stats,on='player_id',rsuffix='_hist')
        # age at same season's edition reference date can differ by a year; median minimizes this ambiguity.
        delta=p['max']-p['min']; valid=(p['count']>=1)&(delta<=1)&(p.season_start-p['median']).between(15,55)
        mask=p.age.isna()&valid
        p.loc[mask,'age']=np.floor(p.loc[mask,'season_start']-p.loc[mask,'median']+0.5)
        p.loc[mask,'age_fill_method']='cross_season_age_estimate';p.loc[mask,'age_fill_reference']='stable_player_id'
    for f in ['height_cm','preferred_foot','nationality_id']:
        if f not in p:continue
        if f=='height_cm':
            x=numeric(p[f]);g=p.assign(x=x).groupby('player_id').x.agg(['min','max','median','count'])
            p=p.join(g,on='player_id',rsuffix=f'_{f}')
            med=p['median']; valid=(p['max']-p['min']<=2)&(p['count']>=1)
            m=x.isna()&valid;p.loc[m,f]=med.loc[m]
            p=p.drop(columns=['min','max','median','count'])
        else:
            fill=p.groupby('player_id')[f].transform(consistent_nonnull)
            m=p[f].isna()&fill.notna();p.loc[m,f]=fill.loc[m]
    stats=p.groupby(['season','age_fill_method'],dropna=False).size().rename('count').reset_index()
    p=p.drop(columns=[c for c in ['dob','dob_parsed','ref_date','candidate_dob','season_start','median','min','max','count'] if c in p])
    return p,stats

def restore_history_tactics(clubs,hist_path,fc24_path,selection):
    root=pd.read_csv(selection); sel=root[(root.event=='selected_team_snapshot') & root.version.between(15,24)].copy()
    dates={(int(r.version),int(r.fifa_update)):str(r.snapshot_date) for r in sel.itertuples()}
    selected={v:upd for v,upd in [(k[0],k[1]) for k in dates]}
    use=['team_id','fifa_version','fifa_update','fifa_update_date']+TAC_COLS
    frames=[]
    if hist_path.exists():
        for chunk in pd.read_csv(hist_path,usecols=lambda c:c in use,low_memory=False,chunksize=150000):
            key=chunk.fifa_version.map(selected)
            chunk=chunk[key.notna() & chunk.fifa_update.eq(key)]
            if len(chunk):frames.append(chunk)
    if fc24_path.exists():
        with zipfile.ZipFile(fc24_path) as z:
            f=next((x for x in z.namelist() if Path(x).name == 'male_teams.csv'),None)
            if f:
                d=pd.read_csv(z.open(f),low_memory=False)
                key=d.fifa_version.map(selected);d=d[key.notna() & d.fifa_update.eq(key)]
                frames.append(d[[x for x in use if x in d]])
    if not frames:return pd.DataFrame(columns=['season','club_id'])
    df=pd.concat(frames,ignore_index=True)
    df['season']=(2000+df.fifa_version-1).astype(str)+'-'+df.fifa_version.astype(str).str[-2:]
    df=df.rename(columns={'team_id':'club_id'})
    res=df[['season','club_id']+[x for x in TAC_COLS if x in df]].drop_duplicates(['season','club_id'])
    return res.rename(columns={a:'tactic_'+SEMANTIC[a]+'_legacy' for a in TAC_COLS if a in res})

def restore_modern_tactics(zippath):
    if not zippath.exists():return pd.DataFrame(columns=['season','club_id'])
    arr=[]
    with zipfile.ZipFile(zippath) as z:
        for v in [25,26,27]:
            name=f'fc{v}_teams.csv'
            if name not in z.namelist():continue
            d=pd.read_csv(z.open(name),low_memory=False)
            keep=['teamid','defensivedepth','buildupplay']+[c for c in d.columns if re.fullmatch('cksupport\\d+',c)]
            keep=[c for c in keep if c in d]
            d=d[keep].rename(columns={'teamid':'club_id','defensivedepth':'tactic_defensive_depth_game','buildupplay':'tactic_build_up_play_game_code',**{c:'tactic_'+c+'_game_code' for c in keep if c.startswith('cksupport')}})
            d['season']=f'{2000+v-1}-{v:02d}'
            arr.append(d)
    return pd.concat(arr,ignore_index=True) if arr else pd.DataFrame(columns=['season','club_id'])

def normalize_tactics(c):
    c=c.copy()
    # Unify labels where semantic meaning is clear; retain numeric codes in separate versioned columns.
    lb='tactic_build_up_play_legacy'
    if lb not in c:c[lb]=pd.NA
    c['tactic_build_up_style']=c[lb].astype('string').str.strip().str.lower().replace({'fast build up':'fast_build_up','slow build up':'slow_build_up','long ball':'long_ball','balanced':'balanced','direct passing':'direct_passing'})
    c['tactic_build_up_style']=c['tactic_build_up_style'].str.replace(' ','_',regex=False)
    game_codes=c.get('tactic_build_up_play_game_code',pd.Series(np.nan,index=c.index))
    c['tactic_build_up_token']=c['tactic_build_up_style'].where(c['tactic_build_up_style'].notna(),game_codes.map(lambda v:('game_code:'+str(int(v))) if pd.notna(v) else pd.NA))
    c['tactic_defensive_style']=c.get('tactic_defensive_style_legacy',pd.Series(pd.NA,index=c.index)).astype('string').str.lower().str.strip().str.replace(' ','_',regex=False)
    # All-null or duplicate legacy compatibility columns are removed after rescuing their information.
    c=c.drop(columns=['formation','build_up_play_raw','def_depth_raw'],errors='ignore')
    # Keep game codes as codes; no proof 1/2/3 share exact historical semantics.
    if 'tactic_defensive_depth_legacy' in c:
        c['tactic_defensive_depth_raw']=numeric(c['tactic_defensive_depth_legacy']).combine_first(numeric(c.get('tactic_defensive_depth_game',pd.Series(np.nan,index=c.index))))
    else:c['tactic_defensive_depth_raw']=numeric(c.get('tactic_defensive_depth_game',pd.Series(np.nan,index=c.index)))
    c['tactic_defensive_depth_version']=np.where(c.fifa_version>=25,'fc25_27_game_code','sofifa_legacy_edition_scale')
    c['tactic_build_up_version']=np.where(c.fifa_version>=25,'fc25_27_game_code','sofifa_legacy_text')
    return c

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',default='output');ap.add_argument('--out',default='output_final');ap.add_argument('--history-teams',default='male_teams.csv');ap.add_argument('--fc24-zip',default='24(1).zip');ap.add_argument('--eafc-zip',default='eafc_db_output.zip');a=ap.parse_args()
    base=Path(a.input);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    p=load_csv(base/'players_2015_2027.csv');c=load_csv(base/'clubs_2015_2027.csv');co=load_csv(base/'coaches_2015_2027.csv');cf=load_csv(base/'club_features_2015_2027.csv')
    p,age_report=fill_player_age(p)
    # Do not ever fill rating/finance/club attributes by copying a different season.
    age_report.to_csv(out/'age_fill_report.csv',index=False,encoding='utf-8-sig')
    # Preserve useful codes and stylized playstyles, discard purely decorative text.
    p=p.drop(columns=[x for x in TEXT_TO_DROP_PLAYER if x in p]);c=c.drop(columns=[x for x in TEXT_TO_DROP_CLUB if x in c]);co=co.drop(columns=['nationality_name'],errors='ignore')
    hist=restore_history_tactics(c,Path(a.history_teams),Path(a.fc24_zip),base/'selected_snapshots.csv')
    modern=restore_modern_tactics(Path(a.eafc_zip))
    c=c.merge(hist,on=['season','club_id'],how='left',validate='one_to_one').merge(modern,on=['season','club_id'],how='left',validate='one_to_one')
    c=normalize_tactics(c)
    # raw existing aliases not removed, to support backwards compatibility; avoid fake cross-edition equalities.
    for field in ['value_eur','wage_eur','release_clause_eur']:
        if field in p:p[field]=numeric(p[field]).mask(numeric(p[field])<0)
    for field in ['club_worth_eur']:
        if field in c:c[field]=numeric(c[field]).mask(numeric(c[field])<=0)
    # Strictly preserve missing age for players without justified inference.
    cf=cf.drop(columns=['age_top11_mean'],errors='ignore')
    # Regenerate original 'top11 by Overall' age summary, making its definition explicit.
    p['_rank']=p.groupby(['season','club_id']).overall.rank(ascending=False,method='first')
    top=p[p._rank<=11].groupby(['season','club_id']).age.mean().rename('age_top11_mean')
    cf=cf.merge(top.reset_index(),on=['season','club_id'],how='left',validate='one_to_one')
    p=p.drop(columns=['_rank'])
    for typ,d in [('players',p),('clubs',c),('coaches',co),('club_features',cf)]:
        keys=['season','club_id'] if typ!='players' else ['season','club_id','player_id']
        assert not d.duplicated(keys).any(), f'{typ} duplicated keys'
        d.to_csv(out/f'{typ}_2015_2027.csv',index=False,encoding='utf-8-sig')
    coverage=pd.DataFrame([{'season':s,'n_players':len(x),'age_filled_pct':round(x.age.notna().mean()*100,2),'remaining_age_missing':int(x.age.isna().sum())} for s,x in p.groupby('season')])
    coverage.to_csv(out/'age_quality_by_season.csv',index=False)
    tactical=['season','fifa_version']+[x for x in c if x.startswith('tactic_')]
    cov=c.groupby('season')[[x for x in c if x.startswith('tactic_')]].agg(lambda s:round(s.notna().mean()*100,1)).reset_index()
    cov.to_csv(out/'tactical_feature_coverage.csv',index=False)
    info={'players_rows':len(p),'clubs_rows':len(c),'player_cols':len(p.columns),'club_cols':len(c.columns),'total_age_missing':int(p.age.isna().sum()),'age_fill_method_counts':p.age_fill_method.value_counts(dropna=False).to_dict(),'tactic_fields':[x for x in c if x.startswith('tactic_')]}
    (out/'processing_summary.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(info,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
