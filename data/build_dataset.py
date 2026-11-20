"""FIFA15--FC27 men's club data integration; run: python build_dataset.py --root .
Requires pandas, numpy. Uses original eafc-db-kit export (not sample audit).
Never treats later-snapshot supplement data as pre-season features.
"""
from __future__ import annotations
import argparse, json, re, zipfile, io, sys, unicodedata
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

LEGACY_DETAIL = {
 'attacking_crossing':'crossing','attacking_finishing':'finishing','attacking_heading_accuracy':'headingaccuracy',
 'attacking_short_passing':'shortpassing','attacking_volleys':'volleys','skill_dribbling':'dribbling',
 'skill_curve':'curve','skill_fk_accuracy':'freekickaccuracy','skill_long_passing':'longpassing',
 'skill_ball_control':'ballcontrol','movement_acceleration':'acceleration','movement_sprint_speed':'sprintspeed',
 'movement_agility':'agility','movement_reactions':'reactions','movement_balance':'balance',
 'power_shot_power':'shotpower','power_jumping':'jumping','power_stamina':'stamina',
 'power_strength':'strength','power_long_shots':'longshots','mentality_aggression':'aggression',
 'mentality_interceptions':'interceptions','mentality_positioning':'positioning','mentality_vision':'vision',
 'mentality_penalties':'penalties','mentality_composure':'composure',
 'defending_marking_awareness':'defensiveawareness','defending_standing_tackle':'standingtackle',
 'defending_sliding_tackle':'slidingtackle','goalkeeping_diving':'gkdiving',
 'goalkeeping_handling':'gkhandling','goalkeeping_kicking':'gkkicking',
 'goalkeeping_positioning':'gkpositioning','goalkeeping_reflexes':'gkreflexes'
}
SUMMARY = ['pace','shooting','passing','dribbling','defending','physic']
POSITION_CODE = dict(enumerate(['GK','SW','RWB','RB','RCB','CB','LCB','LB','LWB','RDM','CDM','LDM','RM','RCM','CM','LCM','LM','RAM','CAM','LAM','RF','CF','LF','RW','RS','ST','LS','LW']))
POSITION_RATINGS = ['ls','st','rs','lw','lf','cf','rf','rw','lam','cam','ram','lm','lcm','cm','rcm','rm','lwb','ldm','cdm','rdm','rwb','lb','lcb','cb','rcb','rb','gk']
PLAYER_META = ['season','fifa_version','snapshot_date','snapshot_source','snapshot_timing','player_id','club_id','gender','name','dob','age','height_cm','weight_kg','nationality_id','nationality_name','preferred_foot','player_positions','club_position','squad_role','club_jersey_number','overall','potential','weak_foot','skill_moves','international_reputation','contract_until','club_joined_date','work_rate','value_eur','wage_eur','release_clause_eur','player_traits','playstyles','rating_source','finance_source','finance_snapshot_date','finance_eligible_preseason','rating_conflict']
PLAYER_COLUMNS = PLAYER_META + SUMMARY + list(LEGACY_DETAIL) + POSITION_RATINGS
CLUB_COLUMNS = ['season','fifa_version','snapshot_date','snapshot_source','snapshot_timing','club_id','gender','club_name','league_id','league_name','league_level','nationality_id','nationality_name','overall','attack','midfield','defence','domestic_prestige','international_prestige','club_worth_raw','club_worth_eur','formation','build_up_play_raw','def_depth_raw','coach_id','manager_name','manager_name_available','source_team_rating']
COACH_COLUMNS = ['season','club_id','coach_id','name','nationality_id','nationality_name','manager_name_available','manager_rating_raw','source']
FEATURE_COLUMNS = ['season','club_id','club_name','snapshot_date','squad_count','overall_mean','overall_top11_mean','overall_top16_mean','overall_top11_std','overall_top11_max','potential_top11_mean','age_top11_mean','height_top11_mean','squad_depth_75','squad_depth_80','squad_depth_85','gk_best_overall','attackers_top5_overall','midfielders_top5_overall','defenders_top5_overall','attackers_finishing_mean','attackers_shot_power_mean','midfielders_vision_mean','midfielders_short_passing_mean','defenders_interceptions_mean','defenders_standing_tackle_mean','gk_diving_best','gk_reflexes_best','pace_top11_mean','shooting_top11_mean','passing_top11_mean','dribbling_top11_mean','defending_top11_mean','physic_top11_mean','club_overall','club_attack','club_midfield','club_defence','international_prestige','domestic_prestige']
META_CODE = ['ID','derive','lookup','raw','supplement','none']

# Men's UEFA Champions League first match dates, supplied by the project team.
# FIFA15 corresponds to the 2014-15 UCL; its opening date was 2014-09-16.
UCL_START_DATES = {
 15:'2014-09-16', 16:'2015-09-15', 17:'2016-09-13',
 18:'2017-09-12', 19:'2018-09-18', 20:'2019-09-17',
 21:'2020-10-20', 22:'2021-09-14', 23:'2022-09-06',
 24:'2023-09-19', 25:'2024-09-17', 26:'2025-09-16',
 27:'2026-09-08',
}

def select_pre_ucl_snapshot(df, version, date_col, events, source):
 """Return (update, ISO date, selection rule, row count).
 Choose the latest dated update STRICTLY before the UCL opening date;
 if none, choose the earliest available dated update. A complete snapshot
 must have enough data to be meaningful: among a date's update candidates,
 choose the update with the most records and then greatest update number.
 Undated groups are used only if no dated groups exist.
 """
 grouped=(df.groupby(['fifa_update',date_col],dropna=False)
          .size().reset_index(name='n'))
 if grouped.empty:raise ValueError(f'No snapshots for FIFA{version} ({source})')
 grouped['parsed_date']=pd.to_datetime(grouped[date_col], errors='coerce')
 grouped['numeric_update']=pd.to_numeric(grouped['fifa_update'],errors='coerce')
 dated=grouped[grouped.parsed_date.notna()].copy()
 if dated.empty:
  chosen=grouped.sort_values(['n','numeric_update'],ascending=[False,False]).iloc[0]
  rule='undated_fallback'
 else:
  kickoff=pd.Timestamp(UCL_START_DATES[version])
  pre=dated[dated.parsed_date.lt(kickoff)]
  if not pre.empty:
   chosen=pre.sort_values(['parsed_date','n','numeric_update'],ascending=[False,False,False]).iloc[0]
   rule='latest_strictly_before_ucl'
  else:
   chosen=dated.sort_values(['parsed_date','n','numeric_update'],ascending=[True,False,False]).iloc[0]
   rule='earliest_available_after_or_on_ucl'
 upd=int(chosen['fifa_update'])
 stamp=chosen.parsed_date.strftime('%Y-%m-%d') if pd.notna(chosen.parsed_date) else ''
 log_event(events,'selected_player_snapshot',version=version,source=source,
           ucl_start_date=UCL_START_DATES[version],fifa_update=upd,
           snapshot_date=stamp,selection_rule=rule,rows=int(chosen['n']),
           available_updates=len(grouped))
 return upd,stamp,rule,int(chosen['n'])

def snapshot_timing(version,dt):
 if not dt:return 'undated_snapshot'
 date=pd.to_datetime(dt,errors='coerce')
 if pd.isna(date):return 'undated_snapshot'
 return 'pre_ucl_start' if date<pd.Timestamp(UCL_START_DATES[version]) else 'post_ucl_start'

def maybe_num(s): return pd.to_numeric(s,errors='coerce')
def value(d,k, default=np.nan): return d[k] if k in d.columns else pd.Series(default,index=d.index)
def blank(cols,n): return pd.DataFrame(index=range(n),columns=cols)
def read_csv_path(p, **kwargs): return pd.read_csv(p,low_memory=False,**kwargs)
def zip_member(zpath,name):
 with zipfile.ZipFile(zpath) as z:
  matches=[n for n in z.namelist() if n==name or n.endswith('/'+name)]
  if len(matches)!=1: raise FileNotFoundError(f'{name} in {zpath} (matches {matches})')
  return pd.read_csv(z.open(matches[0]),low_memory=False)
def ensure_cols(d, columns):
 for c in columns:
  if c not in d.columns: d[c]=np.nan
 return d[columns]
def nonzero(s):
 x=maybe_num(s);return x.where(x.ne(0))
def clean_money(s):
 txt=s.astype('string').str.replace(',','',regex=False).str.replace('€','',regex=False).str.replace('£','',regex=False).str.replace('$','',regex=False).str.strip()
 mult=txt.str.extract(r'([KMB])$',expand=False).map({'K':1e3,'M':1e6,'B':1e9}).fillna(1)
 txt=txt.str.replace(r'[KMB]$','',regex=True)
 return pd.to_numeric(txt,errors='coerce')*mult

def empty_player(n): return pd.DataFrame({c:pd.Series([np.nan]*n,dtype='object') for c in PLAYER_COLUMNS})
def empty_club(n): return pd.DataFrame({c:pd.Series([np.nan]*n,dtype='object') for c in CLUB_COLUMNS})
def get_year(v):return f'{2000+v-1}-{str((2000+v)%100).zfill(2)}'

def log_event(events,typ,**fields):events.append({'event':typ,**fields})

def load_history(root,events):
 p=root/'raw/history/male_players.csv';t=root/'raw/history/male_teams.csv';coach=root/'raw/history/male_coaches.csv'
 if not p.exists() or not t.exists():raise FileNotFoundError('Historical male_players.csv and male_teams.csv required in raw/history')
 players=read_csv_path(p)
 # Historical Player catalog contains ONE update per year in provided archive; never fabricate an earlier one.
 selected={};years=list(range(15,24));pout=[];tout=[];cout=[]
 for version in years:
  x=players[players['fifa_version'].eq(version)].copy()
  if x.empty:log_event(events,'missing_history_version',version=version);continue
  upd,dt,rule,_=select_pre_ucl_snapshot(x,version,'fifa_update_date',events,'sofifa_history')
  selected[version]=(upd,dt)
  x=x[x['fifa_update'].eq(upd)]
  pout.append(standard_history_players(x,version,dt,events))
 del players
 # team full file contains MANY snapshots. read selected update with chunks to avoid huge memory.
 team_by_year={v:[] for v in selected}
 for part in pd.read_csv(t,chunksize=20000,low_memory=False):
  for v,(upd,dt) in selected.items():
   chunk=part[(part['fifa_version']==v)&(part['fifa_update']==upd)]
   if not chunk.empty:team_by_year[v].append(chunk.copy())
 for v in selected:
  upd,dt=selected[v]
  if not team_by_year[v]:
   log_event(events,'missing_exact_team_update',version=v,update=upd,date=dt)
   continue
  tdf=pd.concat(team_by_year[v],ignore_index=True)
  # Require the same update ID as Player; report any date inconsistency.
  exact=tdf[tdf.fifa_update_date.astype(str).eq(dt)]
  if not exact.empty:
   tdf=exact
   team_date=dt
  else:
   team_dates=pd.to_datetime(tdf.fifa_update_date,errors='coerce').dropna()
   team_date=team_dates.min().strftime('%Y-%m-%d') if not team_dates.empty else ''
   log_event(events,'team_snapshot_date_mismatch',version=v,player_date=dt,team_date=team_date,update=upd)
  tdf=tdf.drop_duplicates(subset=['team_id'])
  log_event(events,'selected_team_snapshot',version=v,source='sofifa_history',
            ucl_start_date=UCL_START_DATES[v],fifa_update=upd,
            snapshot_date=team_date,rows=len(tdf),matched_player_update=True,
            exact_date_match=(team_date==dt))
  tout.append(standard_history_teams(tdf,v,team_date))
 if coach.exists():
  cd=read_csv_path(coach).drop_duplicates('coach_id')
  for club in tout:
   matched=club[['season','club_id','coach_id']].copy().merge(cd,how='left',on='coach_id')
   c=pd.DataFrame({'season':matched.season,'club_id':matched.club_id,'coach_id':matched.coach_id,'name':value(matched,'long_name'),'nationality_id':value(matched,'nationality_id'),'nationality_name':value(matched,'nationality_name'),'manager_name_available':value(matched,'long_name').notna(),'source':'sofifa_coaches'})
   cout.append(ensure_cols(c,COACH_COLUMNS))
 return pout,tout,cout

def standard_history_players(x,v,dt,events):
 o=empty_player(len(x));o['season']=get_year(v);o['fifa_version']=v;o['snapshot_date']=dt;o['snapshot_source']='sofifa_history';o['snapshot_timing']=snapshot_timing(v,dt);o['gender']=0
 map_fields={'player_id':'player_id','club_id':'club_team_id','name':'long_name','dob':'dob','age':'age','height_cm':'height_cm','weight_kg':'weight_kg','nationality_id':'nationality_id','nationality_name':'nationality_name','preferred_foot':'preferred_foot','player_positions':'player_positions','club_position':'club_position','club_jersey_number':'club_jersey_number','overall':'overall','potential':'potential','weak_foot':'weak_foot','skill_moves':'skill_moves','international_reputation':'international_reputation','contract_until':'club_contract_valid_until_year','club_joined_date':'club_joined_date','work_rate':'work_rate','value_eur':'value_eur','wage_eur':'wage_eur','release_clause_eur':'release_clause_eur','player_traits':'player_traits'}
 for dst,src in map_fields.items():o[dst]=x[src].to_numpy() if src in x else np.nan
 for c in SUMMARY+list(LEGACY_DETAIL)+POSITION_RATINGS:o[c]=x[c].to_numpy() if c in x else np.nan
 o['squad_role']=o['club_position'].astype('string').str.upper().map({'SUB':'substitute','RES':'reserve'}).fillna('starting_or_other').to_numpy();o['rating_source']='sofifa_history';o['finance_source']='sofifa_history';o['finance_snapshot_date']=dt;o['finance_eligible_preseason']=False;o['rating_conflict']=False
 return o

def standard_history_teams(x,v,dt):
 o=empty_club(len(x));o['season']=get_year(v);o['fifa_version']=v;o['snapshot_date']=dt;o['snapshot_source']='sofifa_history';o['snapshot_timing']=snapshot_timing(v,dt);o['gender']=0
 d={'club_id':'team_id','club_name':'team_name','league_id':'league_id','league_name':'league_name','league_level':'league_level','nationality_id':'nationality_id','nationality_name':'nationality_name','overall':'overall','attack':'attack','midfield':'midfield','defence':'defence','domestic_prestige':'domestic_prestige','international_prestige':'international_prestige','club_worth_eur':'club_worth_eur','coach_id':'coach_id','def_depth_raw':'def_team_depth','build_up_play_raw':'off_build_up_play'}
 for dst,src in d.items():o[dst]=x[src].to_numpy() if src in x else np.nan
 o['source_team_rating']='sofifa_history';return o

def load_fc24(root,events):
 zp=root/'raw/fc24/24.zip';
 if not zp.exists():raise FileNotFoundError('raw/fc24/24.zip missing (the supplied 24(1).zip renamed)')
 p=zip_member(zp,'male_players.csv');t=zip_member(zp,'male_teams.csv');c=zip_member(zp,'male_coaches.csv')
 p=p[p['fifa_version'].eq(24)].copy();t=t[t['fifa_version'].eq(24)].copy()
 if p.empty:raise RuntimeError('FC24 male file did not contain FIFA version 24')
 upd,dt,rule,_=select_pre_ucl_snapshot(p,24,'update_as_of',events,'sofifa_fc24')
 p=p[(p.fifa_update.eq(upd)) & p.update_as_of.astype(str).eq(dt)]
 tt=t[(t['fifa_update'].eq(upd)) & (t['update_as_of'].astype(str).eq(dt))]
 if tt.empty: raise RuntimeError('FC24 Player / Team exact update mismatch')
 log_event(events,'selected_team_snapshot',version=24,source='sofifa_fc24',
           ucl_start_date=UCL_START_DATES[24],fifa_update=upd,
           snapshot_date=dt,rows=len(tt),matched_player_update=True,exact_date_match=True)
 t=tt.drop_duplicates('team_id')
 p=p.rename(columns={'update_as_of':'fifa_update_date'});t=t.rename(columns={'update_as_of':'fifa_update_date'})
 outp=standard_history_players(p,24,str(dt),events);outt=standard_history_teams(t,24,str(dt))
 o=outt[['season','club_id','coach_id']].merge(c.drop_duplicates('coach_id'),on='coach_id',how='left')
 co=ensure_cols(pd.DataFrame({'season':o.season,'club_id':o.club_id,'coach_id':o.coach_id,'name':value(o,'long_name'),'nationality_id':value(o,'nationality_id'),'nationality_name':value(o,'nationality_name'),'manager_name_available':value(o,'long_name').notna(),'source':'sofifa_fc24'}),COACH_COLUMNS)
 return outp,outt,co

def load_raw_eafc(zpath,v,table):return zip_member(zpath,f'fc{v}_{table}.csv')
def load_clean_eafc(zpath,v,table):return zip_member(zpath,f'fc{v}_{table}.csv')
def select_links(rawlinks,clubs,events,version):
 links=rawlinks[['playerid','teamid','position','jerseynumber']].dropna(subset=['playerid','teamid']).copy()
 links=links[links.teamid.isin(clubs.club_id)].drop_duplicates(['playerid','teamid'])
 # Do not arbitrarily assign multi-club players: preserve each link; expose audit.
 multi=links.groupby('playerid').teamid.nunique();log_event(events,'multi_club_links',version=version,players=int((multi>1).sum()))
 return links

def load_eafc(root,v,events):
 zr=root/'raw/eafc/eafc_db_output.zip';zc=root/'raw/eafc/eafc_core_cleaned.zip'
 if not zr.exists():raise FileNotFoundError('eafc_db_output.zip missing')
 p=load_raw_eafc(zr,v,'players');teams=load_raw_eafc(zr,v,'teams');links=load_raw_eafc(zr,v,'teamplayerlinks');leagues=load_raw_eafc(zr,v,'leagues');lt=load_raw_eafc(zr,v,'leagueteamlinks');mgr=load_raw_eafc(zr,v,'manager')
 p=p[p.gender.eq(0)].copy();teams=teams[teams.gender.eq(0)].copy()
 if zc.exists():
  reference=load_clean_eafc(zc,v,'clubs');ids=set(reference.teamid)
  teams=teams[teams.teamid.isin(ids)].copy()
 else:
  # Fallback: women's leagues and national teams must be excluded; require league memberships.
  eligible=set(leagues.loc[leagues.iswomencompetition.eq(0)&leagues.isinternationalleague.eq(0),'leagueid'])
  ids=set(lt.loc[lt.leagueid.isin(eligible),'teamid'])
  teams=teams[teams.teamid.isin(ids)].copy()
  log_event(events,'club_filter_fallback',version=v,clubs=len(teams))
 ls=select_links(links, pd.DataFrame({'club_id':teams.teamid}),events,v)
 p=p.merge(ls,left_on='playerid',right_on='playerid',how='inner',validate='one_to_many',suffixes=('','_link'))
 # A template is a game edition, NOT verified historical snapshot date.
 season=get_year(v);dt='';n=len(p)
 o=empty_player(n);o['season']=season;o['fifa_version']=v;o['snapshot_date']=np.nan;o['snapshot_source']=f'eafc_db_template_fc{v}';o['snapshot_timing']='unverified_template';o['gender']=0
 m={'player_id':'playerid','club_id':'teamid','height_cm':'height','weight_kg':'weight','nationality_id':'nationality','overall':'overallrating','potential':'potential','weak_foot':'weakfootabilitytypecode','skill_moves':'skillmoves','international_reputation':'internationalrep','contract_until':'contractvaliduntil','club_jersey_number':'jerseynumber','club_position':'position'}
 for dst,src in m.items():o[dst]=p[src].to_numpy() if src in p else np.nan
 # Birthdate, join date and preferredposition are proprietary codes. Do NOT pretend they are ISO dates/labels.
 o['dob']=np.nan;o['club_joined_date']=np.nan;o['preferred_foot']=p['preferredfoot'].map({1:'Right',2:'Left'}).to_numpy() if 'preferredfoot' in p else np.nan
 if 'preferredfoot' in p: log_event(events,'foot_code_assumption',version=v, mapping='1=Right,2=Left; confirm via known players')
 for dest,src in LEGACY_DETAIL.items():o[dest]=p[src].to_numpy() if src in p else np.nan
 # Decode EA position IDs; API confirms CM=14/ST=25/LW=27; preserves full detail in source ZIP.
 pos_code=pd.to_numeric(p['position'],errors='coerce')
 o['club_position']=pos_code.map(POSITION_CODE).to_numpy()
 o['squad_role']=pos_code.map({28:'substitute',29:'reserve'}).fillna('starting').to_numpy()
 poscols=[f'preferredposition{i}' for i in range(1,10) if f'preferredposition{i}' in p.columns]
 if poscols:
  decoded=p[poscols].apply(lambda r: ', '.join(dict.fromkeys(POSITION_CODE.get(int(x),'') for x in r if pd.notna(x) and int(x) in POSITION_CODE)),axis=1).to_numpy()
  o['player_positions']=decoded
  primary=p['preferredposition1'].map(POSITION_CODE).to_numpy()
  o['club_position']=o['club_position'].where(o['club_position'].notna(),primary)
 o['rating_source']=f'eafc_template_fc{v}';o['finance_source']=np.nan;o['finance_snapshot_date']=np.nan;o['finance_eligible_preseason']=False;o['rating_conflict']=False
 # original joins may put data in duplicated columns; use explicitly from player+link.
 if 'wage' in p: o['wage_eur']=nonzero(p['wage']).to_numpy();o['finance_source']=o['wage_eur'].map(lambda x: 'eafc_template_fc27_unverified_units' if pd.notna(x) else np.nan)
 if 'releaseclause' in p:o['release_clause_eur']=nonzero(p['releaseclause']).to_numpy()
 # prefer API names/positions (validated by player ID); retain raw preferred-position codes in raw data only.
 league=lt[['teamid','leagueid']].merge(leagues[['leagueid','leaguename']],on='leagueid',how='left').drop_duplicates('teamid')
 tm=teams.merge(league,on='teamid',how='left')
 c=empty_club(len(tm));c['season']=season;c['fifa_version']=v;c['snapshot_date']=np.nan;c['snapshot_source']=f'eafc_db_template_fc{v}';c['snapshot_timing']='unverified_template';c['gender']=0
 tmmap={'club_id':'teamid','club_name':'teamname','league_id':'leagueid','league_name':'leaguename','overall':'overallrating','attack':'attackrating','midfield':'midfieldrating','defence':'defenserating','domestic_prestige':'domesticprestige','international_prestige':'internationalprestige','club_worth_raw':'clubworth','build_up_play_raw':'buildupplay','def_depth_raw':'defensivedepth'}
 for dest,src in tmmap.items():c[dest]=tm[src].to_numpy() if src in tm else np.nan
 c['source_team_rating']=f'eafc_template_fc{v}'
 mg=mgr[mgr.teamid.isin(set(tm.teamid))].drop_duplicates('teamid')
 co=pd.DataFrame({'season':season,'club_id':mg.teamid,'coach_id':mg.managerid,'name':(mg.get('commonname',pd.Series(index=mg.index,dtype=object)).fillna('').astype(str).str.strip()).where(lambda s:s.ne(''))})
 # Recover real manager name from firstname+surname where commonname is missing.
 other=(value(mg,'firstname').fillna('').astype(str)+' '+value(mg,'surname').fillna('').astype(str)).str.strip()
 co['name']=co['name'].fillna(pd.Series(other.to_numpy(),index=co.index)).replace('',np.nan)
 co['manager_name_available']=co.name.notna();co['nationality_id']=value(mg,'nationality').to_numpy();co['nationality_name']=np.nan;co['manager_rating_raw']=value(mg,'starrating').to_numpy();co['source']=f'eafc_template_fc{v}'
 c['coach_id']=c.club_id.map(co.set_index('club_id').coach_id)
 c['manager_name']=c.club_id.map(co.set_index('club_id').name);c['manager_name_available']=c.manager_name.notna()
 log_event(events,'template_loaded',version=v,players=len(o),clubs=len(c),managers=len(co))
 return o,c,ensure_cols(co,COACH_COLUMNS)

def extract_id_url(s):return pd.to_numeric(s.astype('string').str.extract(r'/([0-9]+)(?:/?(?:\?.*)?)$',expand=False),errors='coerce')
def apply_patch(base,patch,cols,source,events,edition, require_rating=True,only_na=True, source_date=None, finance=False):
 if patch is None or patch.empty:return base
 patch=patch.dropna(subset=['player_id']).drop_duplicates('player_id')
 ids=base['player_id'].copy()
 source_fields=list(dict.fromkeys(['player_id','overall']+cols))
 patch=ensure_cols(patch.copy(),source_fields)
 tmp=base[['player_id','overall']].merge(patch,on='player_id',how='left',suffixes=('_base','_patch'),validate='many_to_one')
 valid=tmp['overall_patch'].notna()
 if require_rating:valid &= maybe_num(tmp.overall_base).eq(maybe_num(tmp.overall_patch))
 conflict=(tmp['overall_patch'].notna()&~valid);base.loc[conflict.to_numpy(),'rating_conflict']=True
 filled={}
 for dest in cols:
  if dest not in base.columns:continue
  vals=tmp[dest].to_numpy();mask=valid.to_numpy() & pd.notna(vals)
  if only_na:mask &= base[dest].isna().to_numpy()
  if dest in ['value_eur','wage_eur','release_clause_eur']:
   mask &= pd.to_numeric(pd.Series(vals),errors='coerce').fillna(0).gt(0).to_numpy()
  if dest in ['dob','name','player_positions','playstyles','preferred_foot','work_rate','player_traits','club_joined_date']:
   base[dest]=base[dest].astype(object)
  base.loc[mask,dest]=vals[mask]
  filled[dest]=int(mask.sum())
 if finance:
  any_fin=np.zeros(len(base),dtype=bool)
  for dest in cols:
   if dest in ['value_eur','wage_eur','release_clause_eur']:
    # only mark if a patch value was used; provenance source level if already filled by prior patch
    any_fin |= base[dest].notna().to_numpy() & base['finance_source'].isna().to_numpy()
  base.loc[any_fin,'finance_source']=source
  base.loc[any_fin,'finance_snapshot_date']=source_date
  base.loc[any_fin,'finance_eligible_preseason']=False
 log_event(events,'patch',season=edition,source=source,matched=int(tmp.overall_patch.notna().sum()),rating_conflicts=int(conflict.sum()),filled=filled)
 return base

def api_patch(root,v,events):
 zp=root/'raw/api/ea_fc_data.zip'
 if not zp.exists():return None
 with zipfile.ZipFile(zp) as z:
  target=f'fc{v}/male/players_raw.jsonl'
  if target not in z.namelist():return None
  players=[]
  with z.open(target) as fd:
   for line in fd:
    try:d=json.loads(line)
    except:continue
    stats=d.get('stats') or {};ss=lambda key: (stats.get(key) or {}).get('value')
    ff=d.get('firstName') or '';ll=d.get('lastName') or ''
    a=d.get('alternatePositions') or [];pos=[(d.get('position') or {}).get('shortLabel')]+[i.get('shortLabel') for i in a if isinstance(i,dict)]
    names=[i.get('label') for i in (d.get('playerAbilities') or []) if isinstance(i,dict)]
    row={'player_id':d.get('id'),'overall':d.get('overallRating'),'name':d.get('commonName') or (ff+' '+ll).strip(),'player_positions':', '.join(i for i in pos if i),'preferred_foot':{1:'Right',2:'Left'}.get(d.get('preferredFoot')),'dob':d.get('birthdate'),'playstyles':', '.join(i for i in names if i)}
    for dst,key in zip(SUMMARY,['pac','sho','pas','dri','def','phy']):row[dst]=ss(key)
    players.append(row)
 log_event(events,'api_loaded',version=v,records=len(players));return pd.DataFrame(players)

def fc25_kaggle_patch(root):
 path=root/'raw/fc25/all_players.csv'
 if not path.exists():return None
 d=read_csv_path(path)
 d['player_id']=extract_id_url(d['url']);d['overall']=maybe_num(d['OVR'])
 rename={'Name':'name','Position':'player_positions','play style':'playstyles','PAC':'pace','SHO':'shooting','PAS':'passing','DRI':'dribbling','DEF':'defending','PHY':'physic','Preferred foot':'preferred_foot','Age':'age'}
 return d.rename(columns=rename)

def optional_patch(root,v,events):
 if v==26:
  p=root/'raw/optional/fc26_rovnez.csv'
  if p.exists():
   d=read_csv_path(p);d['player_id']=maybe_num(d['player_id']);d['overall']=maybe_num(d['overall']);return [('fc26_rovnez',d,['age','dob','name','value_eur','wage_eur','release_clause_eur','club_joined_date','work_rate','player_traits']+POSITION_RATINGS,True)]
 if v==25:
  p=root/'raw/optional/fc25_june.csv'
  if p.exists():
   d=read_csv_path(p);d['player_id']=maybe_num(d['player_id']);d['overall']=maybe_num(d['overall_rating']);d['value_eur']=clean_money(d['value']);d['wage_eur']=clean_money(d['wage']);d['release_clause_eur']=clean_money(d['release_clause']);d['name']=value(d,'full_name');d['club_joined_date']=value(d,'club_joined')
   return [('fc25_june_2025-06-03',d,['value_eur','wage_eur','release_clause_eur','name','club_joined_date'],True)]
 return []

def normalize_name(x):
 if pd.isna(x):return ''
 x=unicodedata.normalize('NFKD',str(x)).encode('ascii','ignore').decode('ascii').lower()
 return re.sub(r'[^a-z0-9]+',' ',x).strip()

def fc25_merge_patch(base,root,events):
 """Use SoFIFA Merge only with unique name+club+overall+skill evidence; no fuzzy guesses."""
 file=root/'raw/optional/fc25_merge.csv'
 if not file.exists():return base
 d=read_csv_path(file)
 must={'name','club_id','overall_rating','acceleration','finishing','shot_power','value','wage'}
 if not must.issubset(d.columns):
  log_event(events,'fc25_merge_missing_columns',columns=sorted(must-set(d.columns)));return base
 d['name_key']=d['name'].map(normalize_name)
 d['club_key']=pd.to_numeric(d['club_id'],errors='coerce')
 d['overall_key']=pd.to_numeric(d['overall_rating'],errors='coerce')
 d=d[d.name_key.ne('')].copy()
 for raw,dst in [('acceleration','movement_acceleration'),('finishing','attacking_finishing'),('shot_power','power_shot_power')]:d['source_'+dst]=pd.to_numeric(d[raw],errors='coerce')
 left=base[['player_id','club_id','overall','name','movement_acceleration','attacking_finishing','power_shot_power']].copy()
 left['name_key']=left.name.map(normalize_name);left['club_key']=pd.to_numeric(left.club_id,errors='coerce');left['overall_key']=pd.to_numeric(left.overall,errors='coerce')
 left=left[left.name_key.ne('')]
 candidates=left.merge(d,on=['name_key','club_key','overall_key'],how='inner',suffixes=('_game','_extra'))
 if candidates.empty:
  log_event(events,'fc25_merge',matched=0,reason='no_high_confidence_candidates');return base
 skills=[('movement_acceleration','source_movement_acceleration'),('attacking_finishing','source_attacking_finishing'),('power_shot_power','source_power_shot_power')]
 score=sum(pd.to_numeric(candidates[a],errors='coerce').eq(pd.to_numeric(candidates[b],errors='coerce')) for a,b in skills)
 candidates=candidates.loc[score>=2].copy()
 counts=candidates.groupby('player_id').size()
 candidates=candidates[candidates.player_id.isin(counts[counts.eq(1)].index)].drop_duplicates('player_id')
 if candidates.empty:
  log_event(events,'fc25_merge',matched=0,reason='nonunique_candidates');return base
 patch=pd.DataFrame({'player_id':candidates.player_id,'overall':candidates.overall_key,
 'value_eur':clean_money(candidates.value),'wage_eur':clean_money(candidates.wage),
 'release_clause_eur':clean_money(value(candidates,'release_clause')),
 'name':value(candidates,'full_name')})
 log_event(events,'fc25_merge',matched=len(patch),reason='unique_name_club_overall_and_2of3_skills')
 return apply_patch(base,patch,['value_eur','wage_eur','release_clause_eur','name'],'fc25_sofifa_merge_unverified_date',events,'2024-25',finance=True,source_date='unverified')

def validate_player_club(p,c,events):
 key=c[['season','club_id']].drop_duplicates()
 x=p[['season','club_id']].merge(key,on=['season','club_id'],how='left',indicator=True)
 bad=int(x['_merge'].ne('both').sum())
 log_event(events,'player_club_orphans',count=bad)
 if bad:raise RuntimeError(f'{bad} players linked to missing clubs')

def aggregate(p,clubs):
    """Vectorized club aggregates, avoids per-club Python loops on 200k rows."""
    g=p.copy()
    group=['season','club_id']
    numeric=['overall','potential','age','height_cm']+SUMMARY+['attacking_finishing','power_shot_power','mentality_vision','attacking_short_passing','mentality_interceptions','defending_standing_tackle','goalkeeping_diving','goalkeeping_reflexes']
    for x in numeric:g[x]=pd.to_numeric(g[x],errors='coerce')
    g=g.sort_values(['season','club_id','overall'],ascending=[True,True,False],kind='stable')
    gr=g.groupby(group,observed=True,sort=False)
    agg=gr.agg(squad_count=('player_id','size'),overall_mean=('overall','mean'),squad_depth_75=('overall',lambda x:(x>=75).sum()),squad_depth_80=('overall',lambda x:(x>=80).sum()),squad_depth_85=('overall',lambda x:(x>=85).sum()))
    top11=g.groupby(group,sort=False,observed=True).head(11)
    top16=g.groupby(group,sort=False,observed=True).head(16)
    agg=agg.join(top11.groupby(group,observed=True).agg(overall_top11_mean=('overall','mean'),overall_top11_std=('overall','std'),overall_top11_max=('overall','max'),potential_top11_mean=('potential','mean'),age_top11_mean=('age','mean'),height_top11_mean=('height_cm','mean')))
    agg=agg.join(top16.groupby(group,observed=True).agg(overall_top16_mean=('overall','mean')))
    summ=top11.groupby(group,observed=True)[SUMMARY].mean().rename(columns={k:k+'_top11_mean' for k in SUMMARY})
    agg=agg.join(summ)
    pcode=g['club_position'].astype('string').str.upper().str.strip()
    alt=g['player_positions'].fillna('').astype(str).str.split(',').str[0].str.upper().str.strip()
    pos=pcode.where(pcode.str.fullmatch(r'[A-Z]{1,4}').fillna(False) & ~pcode.isin(['SUB','RES']),alt)
    locs={ 'a':['ST','CF','RW','LW','RF','LF','RS','LS'],
           'm':['CM','CAM','CDM','RM','LM','LAM','RAM','LCM','RCM','LDM','RDM'],
           'd':['CB','LB','RB','LWB','RWB','LCB','RCB'], 'g':['GK']}
    for key,codes in locs.items():
        subset=g[pos.isin(codes)]
        if subset.empty:continue
        if key=='g':
            a=subset.groupby(group,observed=True).agg(gk_best_overall=('overall','max'),gk_diving_best=('goalkeeping_diving','max'),gk_reflexes_best=('goalkeeping_reflexes','max'))
        else:
            top5=subset.groupby(group,sort=False,observed=True).head(5)
            names={'a':('attackers_top5_overall','attackers_finishing_mean','attackers_shot_power_mean'), 'm':('midfielders_top5_overall','midfielders_vision_mean','midfielders_short_passing_mean'), 'd':('defenders_top5_overall','defenders_interceptions_mean','defenders_standing_tackle_mean')}
            n1,n2,n3=names[key]
            vals={'a':['attacking_finishing','power_shot_power'],'m':['mentality_vision','attacking_short_passing'],'d':['mentality_interceptions','defending_standing_tackle']}[key]
            a=top5.groupby(group,observed=True).agg(**{n1:('overall','mean')})
            b=subset.groupby(group,observed=True)[vals].mean()
            a=a.join(b.rename(columns={vals[0]:n2,vals[1]:n3}))
        agg=agg.join(a)
    base=clubs[['season','club_id','club_name','snapshot_date','overall','attack','midfield','defence','international_prestige','domestic_prestige']].rename(columns={'overall':'club_overall','attack':'club_attack','midfield':'club_midfield','defence':'club_defence'})
    out=base.merge(agg.reset_index(),on=group,how='left',validate='one_to_one')
    return ensure_cols(out,FEATURE_COLUMNS)

def docs(root,p,c,m,cf,events):
 q=root/'output';q.mkdir(parents=True,exist_ok=True)
 p.to_csv(q/'players_2015_2027.csv',index=False,encoding='utf-8-sig');c.to_csv(q/'clubs_2015_2027.csv',index=False,encoding='utf-8-sig');m.to_csv(q/'coaches_2015_2027.csv',index=False,encoding='utf-8-sig');cf.to_csv(q/'club_features_2015_2027.csv',index=False,encoding='utf-8-sig')
 for table,df in [('players',p),('clubs',c),('coaches',m),('club_features',cf)]:
  coverage=df.groupby('season',dropna=False).agg(rows=('season','size')).reset_index()
  rows=[{'season':season,'field':field,'non_null_rate':round(float(g[field].notna().mean()),4)} for season,g in df.groupby('season',dropna=False) for field in df.columns]
  long=pd.DataFrame(rows).merge(coverage,on='season');long.insert(0,'table',table);long.to_csv(q/(table+'_coverage.csv'),index=False,encoding='utf-8-sig')
 pd.DataFrame(events).to_json(q/'audit_events.json',force_ascii=False,orient='records',indent=2)
 selected_events=[e for e in events if e.get('event') in ('selected_player_snapshot','selected_team_snapshot','team_snapshot_date_mismatch','missing_exact_team_update')]
 pd.DataFrame(selected_events).to_csv(q/'selected_snapshots.csv',index=False,encoding='utf-8-sig')
 snapshot=p.groupby(['season','fifa_version','snapshot_source','snapshot_date','snapshot_timing'],dropna=False).size().reset_index(name='player_club_rows')
 snapshot.to_csv(q/'snapshot_manifest.csv',index=False,encoding='utf-8-sig')
 # Each row must retain the join key and version; unknown data is NULL, never fabricated.
 if p.duplicated(['season','club_id','player_id']).any():raise ValueError('duplicate player club season rows')
 if c.duplicated(['season','club_id']).any():raise ValueError('duplicate club season rows')
 if sorted(p.fifa_version.dropna().unique().astype(int).tolist())!=list(range(15,28)):
  raise ValueError('incomplete FIFA15-FC27 coverage')
 with (q/'schema_all_columns.txt').open('w',encoding='utf-8') as out:
  for tag,cols in [('PLAYER',PLAYER_COLUMNS),('CLUB',CLUB_COLUMNS),('COACH',COACH_COLUMNS),('CLUB_FEATURE',FEATURE_COLUMNS)]:
   out.write(f'\n[{tag}] {len(cols)} columns\n'+', '.join(cols)+'\n')
 print('\nOUTPUT');print('players',len(p),'clubs',len(c),'coaches',len(m),'club_features',len(cf));print('By season:',p.groupby('season').size().to_dict())


def run(root):
 root=Path(root).resolve();(root/'output').mkdir(parents=True,exist_ok=True);events=[]
 hp,hc,hm=load_history(root,events)
 p24,c24,m24=load_fc24(root,events);hp.append(p24);hc.append(c24);hm.append(m24)
 for v in [25,26,27]:
  p,c,m=load_eafc(root,v,events)
  if v==25:
   d=fc25_kaggle_patch(root)
   if d is not None:p=apply_patch(p,d,['name','player_positions','preferred_foot','playstyles','age']+SUMMARY,f'fc25_nyagami',events,get_year(v))
  else:
   d=api_patch(root,v,events)
   if d is not None:p=apply_patch(p,d,['name','dob','player_positions','playstyles']+SUMMARY,('ea_api_default_fc26_candidate' if v==26 else 'ea_api_fc27'),events,get_year(v))
  for source,d,cols,finance in optional_patch(root,v,events):
   p=apply_patch(p,d,cols,source,events,get_year(v),finance=finance,source_date='2025-06-03' if v==25 else '2025-09-21')
  if v==25:p=fc25_merge_patch(p,root,events)
  # Real DOB obtained from API/optional CSV; native template birthdate is an undecoded integer.
  born=pd.to_datetime(p['dob'],errors='coerce',format='mixed')
  valid=born.notna()
  p['dob']=p['dob'].astype(object)
  p.loc[valid,'dob']=born.loc[valid].dt.strftime('%Y-%m-%d')
  reference=pd.Timestamp(year=2000+v-1,month=9,day=1)
  age=(reference.year-born.dt.year)-((born.dt.month>reference.month)|((born.dt.month==reference.month)&(born.dt.day>reference.day))).astype(int)
  p.loc[p['age'].isna()&valid,'age']=age.loc[p['age'].isna()&valid]
  hp.append(p);hc.append(c);hm.append(m)
 pp=ensure_cols(pd.concat(hp,ignore_index=True),PLAYER_COLUMNS)
 cc=ensure_cols(pd.concat(hc,ignore_index=True),CLUB_COLUMNS)
 mm=ensure_cols(pd.concat(hm,ignore_index=True),COACH_COLUMNS)
 for col in ['player_id','club_id','overall','potential','age','height_cm','weight_kg','nationality_id','weak_foot','skill_moves','international_reputation','contract_until','value_eur','wage_eur','release_clause_eur']+SUMMARY+list(LEGACY_DETAIL):
  pp[col]=pd.to_numeric(pp[col],errors='coerce')
 for col in ['club_id','overall','attack','midfield','defence','league_id','coach_id','international_prestige','domestic_prestige']:
  cc[col]=pd.to_numeric(cc[col],errors='coerce')
 pp=pp.dropna(subset=['club_id']).copy();pp['club_id']=pp.club_id.astype('int64');cc=cc.drop_duplicates(['season','club_id'])
 pp=pp.drop_duplicates(['season','club_id','player_id'])
 # Historical SoFIFA Team exports contain international/national squads; only retain teams
 # referenced in the CLUB slot of male player rows. FC25–27 already filtered via clean club catalog.
 legacy_ids=pp.loc[pp.fifa_version.le(24),['season','club_id']].drop_duplicates().assign(is_club=True)
 before=len(cc)
 cc=cc.merge(legacy_ids,on=['season','club_id'],how='left')
 cc=cc[cc.fifa_version.ge(25)|cc.is_club.fillna(False)].drop(columns='is_club')
 mm=mm.merge(cc[['season','club_id']],on=['season','club_id'],how='inner')
 log_event(events,'excluded_old_national_or_unlinked_teams',count=before-len(cc))
 validate_player_club(pp,cc,events)
 features=aggregate(pp,cc)
 docs(root,pp,cc,mm,features,events)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',default='.');args=ap.parse_args();run(args.root)
