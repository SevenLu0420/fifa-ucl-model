"""UCL match-level builder v2: 2014-15..2026-27, 90-min labels, future fixtures.

Builds historical matches_compact/full (through 2025-26) and separate
2026-27 fixture features. Does not alter the original FIFA/FC files.
Requires pandas, numpy, openpyxl. Run from repository root.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from match_builder.mapping import make_mapping
from match_builder.features import build_team_features, PLAYER_NUM

LEGACY_FIELDS = ['saison','annee','bloc','section','date','heure','domicile',
 'exterieur','buts_domicile','buts_exterieur','score_brut','prolongation',
 'tirs_au_but','stade','spectateurs','arbitre','date_iso']
NEEDED_CSV = ['season','date_iso','home_team','away_team','home_goals_90',
 'away_goals_90','home_goals_final','away_goals_final','extra_time',
 'penalty_shootout','status']


def truth(v):
    if pd.isna(v): return False
    return str(v).strip().casefold() in {'true','1','yes','y'}


def optional_numeric(d, col):
    if col not in d: return pd.Series(np.nan, index=d.index, dtype=float)
    return pd.to_numeric(d[col], errors='coerce')


def date_string(v):
    # openpyxl normally decodes Excel date-formatted cells to datetime; if a
    # workbook stores an unformatted Excel serial, decode it explicitly.
    if pd.isna(v): return np.nan
    if isinstance(v, (int,float,np.integer,np.floating)):
        from openpyxl.utils.datetime import from_excel
        v = from_excel(v)
    return pd.Timestamp(v).strftime('%Y-%m-%d')


def legacy_excel(path):
    from openpyxl import load_workbook
    wb = load_workbook(path, read_only=True, data_only=True)
    rows=[]
    try:
        for ws in wb.worksheets:
            it=ws.iter_rows(values_only=True)
            hdr=[str(x).strip() if x is not None else '' for x in next(it)]
            if hdr != LEGACY_FIELDS:
                raise ValueError(f'Unexpected legacy worksheet {ws.title} columns: {hdr}')
            for excel_row, values in enumerate(it, 2):
                if not any(x is not None for x in values): continue
                d=dict(zip(LEGACY_FIELDS,values))
                rows.append({
                    'season':str(d['saison']).strip(),
                    'match_date':date_string(d['date_iso']),
                    'home_team':str(d['domicile']).strip(),
                    'away_team':str(d['exterieur']).strip(),
                    'home_goals_raw':d['buts_domicile'],
                    'away_goals_raw':d['buts_exterieur'],
                    'home_goals_90':np.nan if truth(d['prolongation']) else d['buts_domicile'],
                    'away_goals_90':np.nan if truth(d['prolongation']) else d['buts_exterieur'],
                    'extra_time':truth(d['prolongation']),
                    'has_penalties':pd.notna(d['tirs_au_but']) and str(d['tirs_au_but']).strip()!='',
                    'penalty_score':d['tirs_au_but'],
                    'status':'played',
                    'stage':'league' if str(d['bloc']).strip().casefold()=='groupes' else 'knockout',
                    'stage_detail':str(d['section']).strip(),
                    'neutral_venue':(str(d['section']).strip().casefold() in {'final','finale'} or
                                      (str(d['saison']).strip()=='2019-20' and
                                       any(k in str(d['section']).strip().casefold() for k in
                                           ('quarter-final','semi-final','quarts de finale','demi-final')))),
                    'source_file':str(path.name), 'source_sheet':ws.title,
                    'source_row':excel_row, 'source_url':'',
                    'score_source':'legacy_120min_if_extra_time',
                })
    finally:
        wb.close()
    return pd.DataFrame(rows)


def supplemental_csv(path):
    df=pd.read_csv(path, low_memory=False, encoding='utf-8-sig')
    missing=set(NEEDED_CSV)-set(df.columns)
    if missing: raise ValueError(f'{path} is missing {sorted(missing)}')
    x=pd.DataFrame(index=df.index)
    x['season']=df.season.astype(str).str.strip()
    x['match_date']=df.date_iso.map(date_string)
    x['home_team']=df.home_team.astype(str).str.strip()
    x['away_team']=df.away_team.astype(str).str.strip()
    x['home_goals_raw']=optional_numeric(df,'home_goals_final')
    x['away_goals_raw']=optional_numeric(df,'away_goals_final')
    x['home_goals_90']=optional_numeric(df,'home_goals_90')
    x['away_goals_90']=optional_numeric(df,'away_goals_90')
    x['extra_time']=df.extra_time.map(truth).astype(bool)
    x['has_penalties']=df.penalty_shootout.map(truth).astype(bool)
    x['penalty_score']=df.apply(lambda r: f"{int(r.penalty_home)}-{int(r.penalty_away)}" if
       pd.notna(r.get('penalty_home')) and pd.notna(r.get('penalty_away')) else np.nan,axis=1)
    x['status']=df.status.astype(str).str.strip().str.lower()
    stage_type=df.get('stage_type',pd.Series('',index=df.index)).fillna('').astype(str)
    stage_raw=df.get('stage',pd.Series('',index=df.index)).fillna('').astype(str)
    league=(stage_type.str.contains('group|league',case=False,regex=True) |
            stage_raw.str.contains('group|league',case=False,regex=True))
    x['stage']=np.where(league,'league','knockout')
    x['stage_detail']=stage_raw
    x['neutral_venue']=(stage_type.str.strip().str.casefold().eq('final') |
                       stage_raw.str.strip().str.casefold().eq('final'))
    x['source_file']=df.get('source_file',pd.Series(path.name,index=df.index)).fillna(path.name)
    x['source_sheet']=''
    x['source_row']=df.index.to_numpy()+2
    x['source_url']=df.get('source_url',pd.Series('',index=df.index)).fillna('')
    x['score_source']='source_csv_90min'
    return x


def load_all_matches(legacy_path, historical_csvs, fixture_csv):
    frames=[legacy_excel(legacy_path)]
    frames.extend(supplemental_csv(p) for p in historical_csvs)
    if fixture_csv is not None: frames.append(supplemental_csv(fixture_csv))
    df=pd.concat(frames,ignore_index=True)
    df['season']=df.season.astype(str)
    for col in ['home_goals_raw','away_goals_raw','home_goals_90','away_goals_90']:
        df[col]=pd.to_numeric(df[col],errors='coerce')
    if not df.status.isin(['played','scheduled']).all():
        raise ValueError('Only played/scheduled status is supported')
    if df[['season','match_date','home_team','away_team']].isna().any().any():
        raise ValueError('Match identity fields cannot be empty')
    if df.duplicated(['season','match_date','home_team','away_team']).any():
        bad=df[df.duplicated(['season','match_date','home_team','away_team'],keep=False)]
        raise ValueError(f'Duplicate fixture keys:\n{bad[["season","match_date","home_team","away_team"]]}')
    if df.loc[df.status=='played',['home_goals_raw','away_goals_raw']].isna().any().any():
        raise ValueError('Played matches must have final scores')
    if df.loc[df.status=='scheduled',['home_goals_raw','away_goals_raw','home_goals_90','away_goals_90']].notna().any().any():
        raise ValueError('Scheduled matches cannot have known goals')
    for col in ['home_goals_raw','away_goals_raw','home_goals_90','away_goals_90']:
        if (df[col].dropna()<0).any():raise ValueError(f'Negative {col}')
        if not ((df[col].dropna()%1)==0).all():raise ValueError(f'Fractional {col}')
    if ((df.home_goals_90>df.home_goals_raw)|(df.away_goals_90>df.away_goals_raw)).any():
        raise ValueError('90-minute goals cannot exceed 120-minute goals')
    df=df.sort_values(['season','match_date','home_team','away_team']).reset_index(drop=True)
    def mid(r):
        raw='|'.join(str(r[c]) for c in ['season','match_date','home_team','away_team'])
        return 'UCL_'+hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]
    df.insert(0,'match_id',df.apply(mid,axis=1))
    df['fifa_version']=df.season.str[:4].astype(int)-1999
    expected=df.season.str[:4].astype(int)
    date_year=pd.to_datetime(df.match_date).dt.year
    if not ((date_year==expected)|(date_year==expected+1)).all():
        raise ValueError('Match date does not fall in its season')
    return df


def apply_90min_corrections(df, path):
    if path is None: return df
    corr=pd.read_csv(path,encoding='utf-8-sig')
    keys=['season','match_date','home_team','away_team']
    scorecols=['home_goals_90','away_goals_90']
    missing=set(keys+scorecols)-set(corr.columns)
    if missing:raise ValueError(f'Correction file missing {sorted(missing)}')
    if corr.duplicated(keys).any():raise ValueError('Duplicate correction keys')
    corr['match_date']=corr.match_date.map(date_string)
    corr['season']=corr.season.astype(str)
    merged=df.merge(corr[keys+scorecols+[x for x in ['home_goals_final_expected','away_goals_final_expected'] if x in corr]].rename(columns={x:x+'_override' for x in scorecols}),
                    on=keys,how='left',validate='one_to_one',indicator=True)
    if int(merged._merge.eq('both').sum())!=len(corr):
        raise ValueError('Some 90-minute corrections did not match a unique legacy match')
    changed=merged._merge.eq('both')
    if (changed & ~merged.extra_time).any():
        raise ValueError('Correction provided for a non-extra-time match')
    # Guard against accidentally applying a correction to another fixture.
    for raw_col, expected_col in [('home_goals_raw','home_goals_final_expected'),
                                  ('away_goals_raw','away_goals_final_expected')]:
        if expected_col in corr.columns:
            expected=merged[expected_col]
            if ((changed & expected.notna()) &
                merged[raw_col].ne(pd.to_numeric(expected,errors='coerce'))).any():
                raise ValueError(f'Original score mismatch in correction: {expected_col}')
    for col in scorecols:
        merged.loc[changed,col]=pd.to_numeric(merged.loc[changed,col+'_override'],errors='raise')
    if merged.loc[changed,scorecols].isna().any().any():
        raise ValueError('Missing correction goals')
    merged.loc[changed,'score_source']='verified_90min_correction'
    return merged.drop(columns=[x+'_override' for x in scorecols]+
        [x for x in ['home_goals_final_expected','away_goals_final_expected'] if x in merged]+['_merge'])


def create_labels(df):
    d=df.copy()
    ok=d.status.eq('played') & d.home_goals_90.notna() & d.away_goals_90.notna()
    diff=d.home_goals_90-d.away_goals_90
    d['label_90min_home_goals']=d.home_goals_90.where(ok)
    d['label_90min_away_goals']=d.away_goals_90.where(ok)
    d['label_90min_goal_diff']=diff.where(ok)
    d['label_90min_wdl']=pd.Series(np.select([diff>0,diff<0],['H','A'],default='D'),index=d.index).where(ok)
    d['label_90min_home_win']=pd.Series(np.where(diff>0,1,0),index=d.index).where(ok)
    d['label_90min_over25']=pd.Series(np.where((d.home_goals_90+d.away_goals_90)>2.5,1,0),index=d.index).where(ok)
    d['both_goals_verified']=ok
    return d


def attach_features(matches, bank):
    all_features=[s for s in bank.columns if s not in {'season','club_id','club_name'}]
    compact_features=[s for s in all_features if not re.match(r'^(gk1|def[1-4]|mid[1-3]|att[1-3])_',s)]
    base=matches.copy()
    for side in ['home','away']:
        key=f'{side}_club_id'
        attach=bank[['season','club_id']+all_features].rename(columns={
            'club_id':key,**{f:f'{side}_{f}' for f in all_features}})
        base=base.merge(attach,on=['season',key],how='left',validate='many_to_one')
    base['both_squads_ready']=base.home_squad_count.notna() & base.away_squad_count.notna()
    base['prediction_ready']=base.both_clubs_mapped & base.both_squads_ready
    base['stage_league']=base.stage.eq('league').astype(int)
    base['stage_knockout']=base.stage.eq('knockout').astype(int)
    return base,compact_features,all_features


def build_outputs(args):
    args.out.mkdir(parents=True,exist_ok=True)
    matches=load_all_matches(args.matches,args.historical_csv,args.future_schedule)
    matches=create_labels(apply_90min_corrections(matches,args.corrections))
    clubs=pd.read_csv(args.clubs,low_memory=False)
    coverage=pd.read_csv(args.coverage,low_memory=False)
    reviewed=pd.read_csv(args.audit_mapping,low_memory=False) if args.audit_mapping and args.audit_mapping.exists() else None
    known=pd.read_csv(args.known_map,low_memory=False) if args.known_map and args.known_map.exists() else None
    mapping=make_mapping(matches,clubs,coverage,reviewed,known)
    mapping.to_csv(args.out/'team_name_mapping.csv',encoding='utf-8-sig',index=False)
    md=mapping[['season','match_team','club_id','mapping_method']]
    canonical=matches.copy()
    for side in ['home','away']:
        canonical=canonical.merge(md.rename(columns={
           'match_team':f'{side}_team','club_id':f'{side}_club_id',
           'mapping_method':f'{side}_mapping_method'}),on=['season',f'{side}_team'],how='left',validate='many_to_one')
    canonical['mapping_collision']=(canonical.home_club_id.notna() & canonical.away_club_id.notna() &
                                    (canonical.home_club_id==canonical.away_club_id))
    canonical['both_clubs_mapped']=(canonical.home_club_id.notna() & canonical.away_club_id.notna() &
                                    ~canonical.mapping_collision)
    canonical.to_csv(args.out/'matches_canonical.csv',encoding='utf-8-sig',index=False)
    needed=pd.concat([canonical[['season','home_club_id']].rename(columns={'home_club_id':'club_id'}),
                      canonical[['season','away_club_id']].rename(columns={'away_club_id':'club_id'})],ignore_index=True)
    needed=needed.dropna().drop_duplicates()
    needed['club_id']=needed.club_id.astype('int64')
    clubs['club_id']=pd.to_numeric(clubs.club_id,errors='coerce').astype('Int64')
    clubs['season']=clubs.season.astype(str)
    clubs_small=clubs.merge(needed,on=['season','club_id'],how='inner',validate='one_to_one')
    player_columns=set(PLAYER_NUM+['season','club_id','player_id','player_positions','club_position','preferred_foot'])
    players=pd.read_csv(args.players,usecols=lambda k:k in player_columns,low_memory=False)
    players['club_id']=pd.to_numeric(players.club_id,errors='coerce').astype('Int64')
    players['season']=players.season.astype(str)
    players=players.merge(needed,on=['season','club_id'],how='inner',validate='many_to_one')
    minimal_cf=clubs_small[['season','club_id']].copy()
    bank,flags=build_team_features(players,clubs_small,minimal_cf)
    bank.to_csv(args.out/'team_feature_bank.csv',index=False,encoding='utf-8-sig')
    flags.to_csv(args.out/'team_feature_quality.csv',index=False,encoding='utf-8-sig')
    full,compact_names,full_names=attach_features(canonical,bank)
    full['in_historical_window']=full.season.str[:4].astype(int)<=2025
    full['eligible_historical']=(full.in_historical_window & full.status.eq('played') &
                                 full.prediction_ready & full.both_goals_verified)
    conditions=[~full.both_clubs_mapped,~full.both_squads_ready,
                full.status.eq('scheduled'),~full.both_goals_verified,
                ~full.in_historical_window]
    reasons=['missing_club_or_mapping_collision','no_valid_xi','scheduled_unplayed',
             'no_verified_90min_score','future_season_reserved_for_simulation']
    full['sample_reason']=np.select(conditions,reasons,default='included_historical')
    full[['match_id','season','match_date','home_team','away_team','status','both_clubs_mapped',
          'both_squads_ready','both_goals_verified','eligible_historical','sample_reason']].to_csv(
             args.out/'sample_status.csv',index=False,encoding='utf-8-sig')
    labels=[k for k in full.columns if k.startswith('label_')]
    metadata=['match_id','season','match_date','fifa_version','home_team','away_team',
              'home_club_id','away_club_id','stage','status','neutral_venue','extra_time',
              'has_penalties','score_source','both_clubs_mapped','both_squads_ready','prediction_ready']
    context=['neutral_venue','stage_league','stage_knockout']
    report={'input_matches':len(full),'by_season':full.groupby('season').size().to_dict(),
       'status_counts':full.status.value_counts().to_dict(),
       'extra_time_matches':int(full.extra_time.sum()),
       'extra_time_with_90min_score':int((full.extra_time & full.both_goals_verified).sum()),
       'matched_matches':int(full.both_clubs_mapped.sum()),
       'usable_historical_matches':int(full.eligible_historical.sum()),
       'future_fixture_rows':int((~full.in_historical_window).sum()),
       'future_prediction_ready':int((~full.in_historical_window & full.prediction_ready).sum()),
       'unmatched_team_seasons':mapping.loc[mapping.status=='UNMATCHED',['season','match_team']].to_dict('records'),
       'sample_reason_counts':full.sample_reason.value_counts().to_dict(),
       'feature_sets':{}}
    for kind, names in [('compact',compact_names),('full',full_names)]:
        features=[f'{side}_{f}' for side in ['home','away'] for f in names]+context
        dfcolumns=list(dict.fromkeys(metadata+features+labels))
        past=full.loc[full.eligible_historical,dfcolumns].copy()
        future=full.loc[~full.in_historical_window,dfcolumns].copy()
        if future.empty and args.future_schedule:raise ValueError('Future schedule unexpectedly empty')
        past.to_csv(args.out/f'matches_{kind}.csv',index=False,encoding='utf-8-sig')
        future.to_csv(args.out/f'fixtures_2026_27_{kind}.csv',index=False,encoding='utf-8-sig')
        (args.out/f'{kind}_feature_columns.json').write_text(json.dumps(features,indent=2),encoding='utf-8')
        report['feature_sets'][kind]={'historical_rows':len(past),'future_rows':len(future),
                                      'historical_predictors':len(features),
                                      'future_predictable_rows':int(future.prediction_ready.sum())}
        half=(len(features)-len(context))//2
        assert features[:half]==['home_'+f for f in names]
        assert features[half:2*half]==['away_'+f for f in names]
        print(f'{kind}: historical={len(past)}, 2026-27 fixtures={len(future)}, predictors={len(features)}')
    (args.out/'build_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Done: {len(full)} canonical; {report["usable_historical_matches"]} historical labeled and squad-ready')
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--matches',type=Path,required=True,help='Baptiste 2015-16..2024-25 legacy XLSX')
    ap.add_argument('--historical-csv',type=Path,action='append',default=[],help='Add 2014-15 and 2025-26 CSV, repeat twice')
    ap.add_argument('--future-schedule',type=Path,help='2026-27 schedule incl played/scheduled fixtures')
    ap.add_argument('--corrections',type=Path,help='Verified 90-minute score corrections for legacy extra-time games')
    ap.add_argument('--players',type=Path,required=True)
    ap.add_argument('--clubs',type=Path,required=True)
    ap.add_argument('--coverage',type=Path,required=True)
    ap.add_argument('--audit-mapping',type=Path)
    ap.add_argument('--known-map',type=Path,help='Repo reference ucl_25_27_verified_mapping.csv')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    build_outputs(args)


if __name__=='__main__':main()
