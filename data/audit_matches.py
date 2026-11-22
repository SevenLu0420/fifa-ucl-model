from __future__ import annotations
import argparse, collections, csv, datetime as dt, difflib, json, math, re, unicodedata
from pathlib import Path
from openpyxl import load_workbook

FIELDS = ['saison','annee','bloc','section','date','heure','domicile','exterieur','buts_domicile','buts_exterieur','score_brut','prolongation','tirs_au_but','stade','spectateurs','arbitre','date_iso']
MISSING_STRINGS={'','nan','none','null','n/a','na','-','?'}

def missing(v):
    return v is None or (isinstance(v,float) and math.isnan(v)) or (isinstance(v,str) and v.strip().lower() in MISSING_STRINGS)

def norm(s):
    s=unicodedata.normalize('NFKD',str(s or '')).encode('ascii','ignore').decode().lower()
    s=re.sub(r'\b(fc|cf|afc|fk|sc|ac|cfc|club|football|futbol|futebol|soccer)\b',' ',s)
    return re.sub(r'[^a-z0-9]','',s)

def csv_write(path, rows, cols):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f, fieldnames=cols, extrasaction='ignore');w.writeheader();w.writerows(rows)

def csv_read(path):
    if not path or not path.exists():return []
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def date_key(v):
    if isinstance(v,(int,float)):
        try:return (dt.date(1899,12,30)+dt.timedelta(days=int(v))).isoformat()
        except (OverflowError,ValueError):return str(v)
    if isinstance(v,(dt.date,dt.datetime)):return v.strftime('%Y-%m-%d')
    return str(v or '').strip()

def parse_pen(s):
    if missing(s): return None
    parts=re.findall(r'\d+',str(s))
    return (int(parts[0]),int(parts[1])) if len(parts)==2 else None

def load_xlsx(path):
    wb = load_workbook(filename=str(path), read_only=True, data_only=True)
    out = []
    sheet_meta = []
    try:
        for sheet in wb.worksheets:
            rows = sheet.iter_rows(values_only=True)
            header_values = next(rows, None)
            if header_values is None:
                continue
            header = [str(v or '').strip() for v in header_values]
            if header != FIELDS:
                raise ValueError(f'Sheet {sheet.title}: unexpected columns {header}; expected {FIELDS}')
            n = 0
            for row_idx, vals in enumerate(rows, start=2):
                if all(missing(v) for v in vals):
                    continue
                r = dict(zip(FIELDS, vals))
                r['source_sheet'] = sheet.title
                r['excel_row'] = row_idx
                r['match_date_iso'] = date_key(r['date_iso'])
                r['fifa_season'] = str(r['saison']) if re.match(r'^\d{4}-\d{2}$', str(r['saison'])) else ''
                out.append(r)
                n += 1
            sheet_meta.append({'sheet': sheet.title, 'matches': n})
    finally:
        wb.close()
    return out, sheet_meta

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matches',type=Path,required=True)
    p.add_argument('--clubs',type=Path,default=None)
    p.add_argument('--coverage',type=Path,default=None)
    p.add_argument('--players',type=Path,default=None)
    p.add_argument('--club-features',type=Path,default=None)
    p.add_argument('--coaches',type=Path,default=None)
    p.add_argument('--out',type=Path,default=Path('match_audit'))
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    matches,sheets=load_xlsx(a.matches)
    clubs=csv_read(a.clubs);cover=csv_read(a.coverage)
    issues=[]
    def flag(r,kind,severity,details):
        issues.append({'sheet':r.get('source_sheet',''),'excel_row':r.get('excel_row',''),'season':r.get('saison',''),'date':r.get('match_date_iso',''),'home':r.get('domicile',''),'away':r.get('exterieur',''),'issue':kind,'severity':severity,'details':details})
    grouped=collections.defaultdict(list)
    for r in matches:grouped[r['saison']].append(r)
    season_stats=[]
    scores=[];keys=collections.defaultdict(list)
    for s,rr in sorted(grouped.items()):
        stages=collections.Counter(str(x['bloc']) for x in rr)
        clubs_this=set(str(x[z]) for x in rr for z in ('domicile','exterieur'))
        season_stats.append({'season':s,'fifa_version':str(int(s[:4])+1),'matches':len(rr),'teams':len(clubs_this),'group_or_league':stages['groupes'],'knockout':stages['elimination'],'missing_scores':sum(missing(x['buts_domicile']) or missing(x['buts_exterieur']) for x in rr),'penalty_matches':sum(not missing(x['tirs_au_but']) for x in rr),'extra_time_matches':sum(x['prolongation'] is True for x in rr)})
        for r in rr:
            key=(s,r['match_date_iso'],norm(r['domicile']),norm(r['exterieur']))
            keys[key].append(r)
            if missing(r['domicile']) or missing(r['exterieur']):flag(r,'missing_team','error','missing home or away team')
            if norm(r['domicile'])==norm(r['exterieur']):flag(r,'same_home_away','error','home and away identical')
            if missing(r['match_date_iso']):flag(r,'missing_date','error','date_iso empty')
            if str(r['source_sheet'])!=str(r['saison']):flag(r,'sheet_season_disagreement','error','sheet and saison disagree')
            hg,ag=r['buts_domicile'],r['buts_exterieur']
            if not (isinstance(hg,(float,int)) and isinstance(ag,(float,int))):flag(r,'missing_or_invalid_score','error',f'{hg} vs {ag}')
            else:
                if hg<0 or ag<0 or hg!=int(hg) or ag!=int(ag):flag(r,'invalid_goal_count','error',f'{hg} vs {ag}')
                scores.append({'season':s,'home_goals':int(hg),'away_goals':int(ag),'goal_difference':int(hg-ag),'outcome':'H' if hg>ag else 'A' if hg<ag else 'D','stage':r['bloc']})
                raw=str(r['score_brut'] or '')
                pair=re.search(r'(\d+)\s*[–—-]\s*(\d+)',raw)
                if pair and (int(pair.group(1)),int(pair.group(2)))!=(int(hg),int(ag)):
                    flag(r,'raw_score_disagreement','error',f'raw={raw}; structured={hg}-{ag}')
            if not missing(r['tirs_au_but']) and parse_pen(r['tirs_au_but']) is None:flag(r,'unparseable_penalties','warning',str(r['tirs_au_but']))
            if not missing(r['tirs_au_but']) and r['prolongation'] is not True:flag(r,'penalties_without_extra_time','warning',str(r['tirs_au_but']))
            if isinstance(r['spectateurs'],(int,float)) and r['spectateurs']<0:flag(r,'negative_attendance','error',str(r['spectateurs']))
            if not missing(r['heure']) and not re.search(r'^\d{1,2}:\d{2}',str(r['heure'])):flag(r,'irregular_time','warning',str(r['heure']))
    for key,rr in keys.items():
        if len(rr)>1:
            for r in rr:flag(r,'duplicate_season_date_teams','error',f'{len(rr)} rows')
    missing_stats=[]
    for field in FIELDS:
        n=sum(missing(r[field]) for r in matches)
        missing_stats.append({'field':field,'missing':n,'total':len(matches),'missing_rate':round(n/len(matches),5) if matches else None,'examples': ' | '.join(str(r[field])[:60] for r in matches if not missing(r[field]))[:180]})
    match_clubs=collections.defaultdict(lambda:{'n':0,'dates':[]})
    for r in matches:
        for side in ['domicile','exterieur']:
            key=(r['saison'],str(r[side]));match_clubs[key]['n']+=1;match_clubs[key]['dates'].append(r['match_date_iso'])
    # Lookup via *verified* UEFA reference FIRST; then direct game names.
    covers=collections.defaultdict(list)
    for c in cover:
        covers[(str(c.get('season','')),norm(c.get('uefa_team','')))].append(c)
    club_by_season=collections.defaultdict(list)
    for c in clubs:
        club_by_season[str(c.get('season',''))].append(c)
    mappings=[]
    for (season,match_name),pinfo in sorted(match_clubs.items()):
        fifa_season=season
        uefa_name_aliases={'Inter Milan':'Inter'}
        refs=covers.get((season,norm(uefa_name_aliases.get(match_name,match_name))),[])
        valid_ref=[c for c in refs if str(c.get('club_id','')).strip() not in {'','nan','None'}]
        game=club_by_season.get(fifa_season,[])
        id_to_clubs=collections.defaultdict(list)
        for c in game:id_to_clubs[str(c['club_id']).removesuffix('.0')].append(c)
        ref_match=[]
        for c in valid_ref:ref_match+=id_to_clubs.get(str(c['club_id']).removesuffix('.0'),[])
        ref_match=list({str(x['club_id']):x for x in ref_match}.values())
        exact=[x for x in game if norm(x.get('club_name'))==norm(match_name)]
        matched=None;method='UNRESOLVED'
        if len(ref_match)==1:matched=ref_match[0];method='verified_uefa_club_id'
        elif len(exact)==1:matched=exact[0];method='unique_exact_normalized'
        elif refs and not valid_ref:method='known_ucl_coverage_missing'
        pool=sorted(game,key=lambda x:difflib.SequenceMatcher(None,norm(match_name),norm(x.get('club_name'))).ratio(),reverse=True)[:3]
        suggestions=' | '.join(f"{x.get('club_name')} [id={x.get('club_id')}, sim={difflib.SequenceMatcher(None,norm(match_name),norm(x.get('club_name'))).ratio():.2f}]" for x in pool)
        mappings.append({'season':season,'fifa_season':fifa_season,'fifa_version':str(int(season[:4])+1),'match_team':match_name,'match_count':pinfo['n'],'mapping_status':'MATCHED' if matched else 'REVIEW','match_method':method,'club_id':matched.get('club_id','') if matched else '','club_name':matched.get('club_name','') if matched else '','verified_reference_team':valid_ref[0].get('uefa_team','') if valid_ref else '','suggested_candidates_NOT_AUTOMATIC':suggestions})
    # Audited stable EA club IDs for common naming variants; never fuzzy auto-map.
    # All IDs still must exist in the selected season to be accepted.
    VERIFIED_ALIASES={
       'Bayern Munich':'21','Inter Milan':'44','CSKA Moscow':'315',
       'Lyon':'66','Olympiacos':'280','Zenit Saint Petersburg':'100769',
       'Copenhagen':'819','Basel':'896','Marseille':'219',
       'Gent':'674','Legia Warsaw':'1871','Anderlecht':'229',
       'Spartak Moscow':'100767','Genk':'673','Slavia Prague':'266',
       'Braga':'1896','Sparta Prague':'267',
    }
    for m in mappings:
        if m['mapping_status']=='MATCHED':continue
        wanted=VERIFIED_ALIASES.get(m['match_team'])
        if not wanted:continue
        possible=[c for c in club_by_season.get(m['fifa_season'],[]) if str(c.get('club_id','')).removesuffix('.0')==wanted]
        if len(possible)==1:
            m.update(mapping_status='MATCHED',match_method='curated_stable_ea_id',club_id=possible[0]['club_id'],club_name=possible[0]['club_name'])
    # Conservative cross-season ID propagation. An identical match-side name already
    # verified in another season is accepted only when the SAME club ID exists in the
    # current season's game table. Does not resolve actually absent clubs.
    known_ids=collections.defaultdict(set)
    for m in mappings:
        if m['mapping_status']=='MATCHED':known_ids[norm(m['match_team'])].add(str(m['club_id']).removesuffix('.0'))
    for m in mappings:
        if m['mapping_status']=='MATCHED':continue
        ids=known_ids.get(norm(m['match_team']),set())
        if len(ids)!=1:continue
        id0=next(iter(ids))
        existing=[c for c in club_by_season.get(m['fifa_season'],[]) if str(c.get('club_id','')).removesuffix('.0')==id0]
        if len(existing)==1:
            m.update(mapping_status='MATCHED',match_method='verified_cross_season_club_id',club_id=existing[0]['club_id'],club_name=existing[0]['club_name'])
    match_index={(r['season'],r['match_team']):r for r in mappings}
    match_coverage=[]
    for s,rr in sorted(grouped.items()):
        n_both=sum(match_index[(s,str(r['domicile']))]['mapping_status']=='MATCHED' and match_index[(s,str(r['exterieur']))]['mapping_status']=='MATCHED' for r in rr)
        match_coverage.append({'season':s,'matches':len(rr),'both_teams_matched':n_both,'one_or_both_unmatched':len(rr)-n_both,'match_join_rate':round(n_both/len(rr),4)})
    result_stats=dict(collections.Counter(x['outcome'] for x in scores))
    severe=collections.Counter(f['issue'] for f in issues)
    summary={'input':str(a.matches),'total_matches':len(matches),'seasons':len(sheets),'season_range':[sheets[0]['sheet'],sheets[-1]['sheet']] if sheets else [],'sheets':sheets,'unique_team_seasons':len(mappings),'mapping_matched':sum(x['mapping_status']=='MATCHED' for x in mappings),'mapping_need_review':sum(x['mapping_status']=='REVIEW' for x in mappings),'match_level_both_clubs_matched':sum(x['both_teams_matched'] for x in match_coverage),'match_level_missing_club':sum(x['one_or_both_unmatched'] for x in match_coverage),'outcomes':result_stats,'raw_issue_counts':dict(severe),'extra_time_count':sum(x['prolongation'] is True for x in matches),'penalty_count':sum(not missing(x['tirs_au_but']) for x in matches),'important_notes':['2019-20 knockout single-leg format and 2024-25 league format require stage-aware processing','Do not treat attendance/referee/stadium from match record as necessarily pre-match available','Penalty shootout must be separate from match score / WDL label','Match-file seasons do not include 2014-15 or 2025-26/2026-27 unless provided separately']}
    csv_write(a.out/'season_summary.csv',season_stats,list(season_stats[0]))
    csv_write(a.out/'field_missingness.csv',missing_stats,list(missing_stats[0]))
    csv_write(a.out/'data_issues.csv',issues,['sheet','excel_row','season','date','home','away','issue','severity','details'])
    csv_write(a.out/'team_name_mapping_review.csv',mappings,list(mappings[0]))
    csv_write(a.out/'match_join_coverage.csv',match_coverage,list(match_coverage[0]))
    csv_write(a.out/'match_scores_audit.csv',scores,list(scores[0]))
    # Optional, streaming per-column missingness and cross-table integrity audit
    # for FIFA data (does not load the ~234k-row player file into memory).
    def audit_csv_columns(path, tag):
        if path is None or not path.exists():return
        columns=collections.defaultdict(lambda: [0,0])
        nrows=0
        with path.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                nrows+=1;season=row.get('season','')
                for col,value in row.items():
                    c=columns[(season,col)];c[1]+=1
                    if missing(value):c[0]+=1
        out_rows=[{'season':s,'field':field,'missing':cnt[0],'total':cnt[1],'missing_rate':round(cnt[0]/cnt[1],5)} for (s,field),cnt in sorted(columns.items())]
        csv_write(a.out/f'{tag}_field_missingness_by_season.csv',out_rows,['season','field','missing','total','missing_rate'])
        summary[f'{tag}_audited_rows']=nrows

    if a.clubs and a.clubs.exists():audit_csv_columns(a.clubs,'clubs')
    if a.club_features and a.club_features.exists():audit_csv_columns(a.club_features,'club_features')
    if a.coaches and a.coaches.exists():audit_csv_columns(a.coaches,'coaches')
    if a.players and a.players.exists():
        audit_csv_columns(a.players,'players')
        seen=set();squads=collections.defaultdict(lambda:{'players':0,'gk':0,'missing_position':0,'missing_overall':0,'missing_age':0})
        player_issues=collections.Counter()
        existing_clubs={(c.get('season',''),str(c.get('club_id','')).removesuffix('.0')) for c in clubs}
        with a.players.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                season=row.get('season','');cid=str(row.get('club_id','')).removesuffix('.0');pid=row.get('player_id','')
                key=(season,pid,cid)
                if key in seen:player_issues['duplicate_season_player_club']+=1
                seen.add(key)
                if existing_clubs and (season,cid) not in existing_clubs:player_issues['missing_foreign_club']+=1
                sq=squads[(season,cid)];sq['players']+=1
                position=str(row.get('player_positions') or '')
                if re.search(r'\bGK\b',position) or row.get('club_position')=='GK':sq['gk']+=1
                if missing(position):sq['missing_position']+=1
                val=row.get('overall')
                if missing(val):sq['missing_overall']+=1
                else:
                    try:
                        if not 0 <= float(val) <= 99:player_issues['overall_out_of_range']+=1
                    except ValueError:player_issues['overall_unparseable']+=1
                age=row.get('age')
                if missing(age):sq['missing_age']+=1
                else:
                    try:
                        if not 12 <= float(age) <= 55:player_issues['age_implausible']+=1
                    except ValueError:player_issues['age_unparseable']+=1
        squad_rows=[]
        for (season,cid),sq in sorted(squads.items()):
            squad_rows.append({'season':season,'club_id':cid,**sq,'no_goalkeeper':int(sq['gk']==0),'under_11_players':int(sq['players']<11)})
        csv_write(a.out/'squad_integrity.csv',squad_rows,list(squad_rows[0]) if squad_rows else ['season','club_id','players','gk','missing_position','missing_overall','missing_age','no_goalkeeper','under_11_players'])
        summary['player_integrity_issues']=dict(player_issues)
        summary['clubs_without_GK']=sum(x['no_goalkeeper'] for x in squad_rows)
        summary['clubs_with_fewer_than_11_players']=sum(x['under_11_players'] for x in squad_rows)

    (a.out/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
