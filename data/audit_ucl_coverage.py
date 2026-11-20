"""UEFA CL main-stage club coverage, FIFA15–FC27.
Live OpenFootball source for 2014–15 through 2023–24; reviewed 36-club mapping for 2024–27.
Never automatically infer 'not in FIFA game' from fuzzy string mismatch.
For disconnected machines: put OpenFootball `cl.txt` files in fixtures/<season>/cl.txt.
"""
from __future__ import annotations
import argparse, io, re, unicodedata, urllib.request, zipfile
from difflib import SequenceMatcher
from pathlib import Path
import pandas as pd

BASE='https://raw.githubusercontent.com/openfootball/champions-league/master/{season}/cl.txt'
ALIASES={
 'Inter':'Inter','FC Internazionale Milano':'Inter','Internazionale':'Inter','Lombardia FC':'Inter',
 'AC Milan':'Milan','Milano FC':'Milan','Bergamo Calcio':'Atalanta','Atalanta BC':'Atalanta',
 'Bor. Mönchengladbach':'Borussia Mönchengladbach','Bayern München':'Bayern München',
 'Paris Saint-Germain FC':'Paris Saint-Germain','Paris':'Paris Saint-Germain',
 'FC København':'FC Copenhagen','FC Copenhagen':'FC Copenhagen','FC Kobenhavn':'FC Copenhagen',
 'Bodø/Glimt':'Bodø/Glimt','FK Bodø/Glimt':'Bodø/Glimt',
 'Crvena Zvezda':'Red Star Belgrade','FK Crvena Zvezda':'Red Star Belgrade','Red Star Belgrade':'Red Star Belgrade',
 'Slovan Bratislava':'Slovan Bratislava','ŠK Slovan Bratislava':'Slovan Bratislava',
 'RB Salzburg':'Red Bull Salzburg','FC Salzburg':'Red Bull Salzburg','Salzburg':'Red Bull Salzburg',
 'Dinamo Kiev':'Dynamo Kyiv','Dynamo Kiev':'Dynamo Kyiv','Dynamo Kyiv':'Dynamo Kyiv',
 'Zenit St. Petersburg':'Zenit St Petersburg','Zenit St. Petersburgh':'Zenit St Petersburg',
 'Shakhtar Donetsk':'Shakhtar Donetsk','FC Shakhtar Donetsk':'Shakhtar Donetsk',
 'FC Basel 1893':'Basel','FC Basel':'Basel','Basel 1893':'Basel',
 'Sporting CP':'Sporting CP','Sporting Clube de Portugal':'Sporting CP',
 'Olympiakos Piraeus':'Olympiacos','Olympiacos':'Olympiacos',
 'APOEL Nikosia':'APOEL','APOEL Nicosia':'APOEL',
 'Dinamo Zagreb':'Dinamo Zagreb','GNK Dinamo Zagreb':'Dinamo Zagreb',
 'AS Roma':'Roma','SSC Napoli':'Napoli','AFC Ajax':'Ajax','PSV Eindhoven':'PSV',
 'Bayer Leverkusen':'Bayer Leverkusen','Borussia Dortmund':'Borussia Dortmund',
 'Atlético Madrid':'Atletico Madrid','Atlético de Madrid':'Atletico Madrid',
 'Real Madrid CF':'Real Madrid','Real Madrid':'Real Madrid',
 'FC Barcelona':'Barcelona','Arsenal FC':'Arsenal','Liverpool FC':'Liverpool',
 'Chelsea FC':'Chelsea','Celtic FC':'Celtic','Tottenham Hotspur':'Tottenham Hotspur',
 'Malmö FF':'Malmo','Malmö':'Malmo','Maccabi Tel Aviv':'Maccabi Tel Aviv',
 'PFC Ludogorets Razgrad':'Ludogorets','Ludogorets Razgrad':'Ludogorets',
 'RSC Anderlecht':'Anderlecht','Racing Club de Lens':'Lens',
 'FC Porto':'Porto','SL Benfica':'Benfica',
 'Juventus FC':'Juventus','Galatasaray':'Galatasaray',
 'Manchester United':'Manchester United','Manchester City':'Manchester City',
 'NK Maribor':'Maribor','Bor. Mönchengladbach':'Borussia Monchengladbach',
 'FC Schalke 04':'Schalke 04','Schalke':'Schalke 04',
 'Beşiktaş':'Besiktas','BATE Borisov':'BATE',
 'CSKA Moskva':'CSKA Moscow','CSKA Moscow':'CSKA Moscow',
 'Lokomotiv Moskva':'Lokomotiv Moscow','Lokomotiv Moscow':'Lokomotiv Moscow',
 'VfL Wolfsburg':'Wolfsburg','Young Boys':'Young Boys','BSC Young Boys':'Young Boys',
 'Copenhagen':'FC Copenhagen','Rangers FC':'Rangers','Celtic FC':'Celtic',
 'Red Bull Leipzig':'RB Leipzig','RB Leipzig':'RB Leipzig',
 'Club Brugge KV':'Club Brugge','KRC Genk':'Genk','Royal Antwerp FC':'Royal Antwerp',
 'FK Astana':'Astana','Pafos FC':'Pafos','Kairat Almaty':'Kairat',
}
REVIEWED_CLUB_IDS={
 'Athletic Club':448, 'Bayer Leverkusen':32, 'Olympiakos Piraeus':280,
 'Beşiktaş':327, 'AEK Athen':278, '1899 Hoffenheim':10029,
 'Lille OSC':65, 'Lazio Roma':46, 'SS Lazio':46,
 'Liverpool FC':9, 'Paris Saint-Germain':73, 'Paris Saint-Germain FC':73,
 '1. FC Union Berlin':1831, 'Feyenoord Rotterdam':246,
 'Real Sociedad de Fútbol':457, 'Royal Antwerp FC':230,
 'Sporting Clube de Braga':1896, 'Sport Lisboa e Benfica':234,
 'FC København':819, 'Stade Rennais':74,
 'FC Internazionale Milano':44,
}
DROP={'fc','cf','ac','afc','as','sc','sk','fk','nk','rsc','vfl','sv','pfk','pfc','rc','football','club','the','de','calcio','sfc'}
def norm(s):
    s=ALIASES.get(str(s).strip(),str(s).strip())
    s=unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode().lower()
    toks=re.findall(r'[a-z0-9]+',s.replace('&',' and '))
    return ' '.join(t for t in toks if t not in DROP)

def extract_group_teams(txt):
    capture=False; teams=set()
    for ln in txt.splitlines():
        line=ln.strip();low=line.lower()
        if line.startswith('▪'):
            if re.search(r'\bgroup(?:\s+[a-h]\b|[, ])|\bleague[, ]',low):capture=True
            elif re.search(r'round of|knockout|quarter|semi|final|play.off',low):capture=False
        if not capture:continue
        m=re.search(r'(.+?)\s+\(([A-Z]{3})\)\s+v\s+(.+?)\s+\(([A-Z]{3})\)',line)
        if not m:continue
        a=re.sub(r'^\s*\d{1,2}:\d{2}\s*','',m.group(1)).strip()
        b=m.group(3).strip()
        teams.update([a,b])
    return sorted(teams)

def fixtures_for_season(season,root,offline=False):
    f=root/season/'cl.txt'
    if not f.exists() and not offline:
        request=urllib.request.Request(BASE.format(season=season),headers={'User-Agent':'Mozilla/5.0 (FIFA research / noncommercial)'})
        with urllib.request.urlopen(request,timeout=25) as q:content=q.read().decode('utf-8-sig')
        f.parent.mkdir(parents=True,exist_ok=True);f.write_text(content,encoding='utf-8')
    if not f.exists():return None
    return extract_group_teams(f.read_text(encoding='utf-8-sig'))

def combined_raw_team_catalog(history,fc24,eafc):
    arr=[]
    if history.exists():
        # All historical game updates, not merely the chosen season opening snapshot.
        for d in pd.read_csv(history,usecols=['fifa_version','team_id','team_name'],chunksize=150000,low_memory=False):
            arr.append(d.rename(columns={'team_id':'club_id','team_name':'club_name'}))
    if fc24.exists():
        with zipfile.ZipFile(fc24) as z:
            f=next((x for x in z.namelist() if Path(x).name=='male_teams.csv'),None)
            if f:
                d=pd.read_csv(z.open(f),usecols=['fifa_version','team_id','team_name'],low_memory=False)
                arr.append(d.rename(columns={'team_id':'club_id','team_name':'club_name'}))
    if eafc.exists():
        with zipfile.ZipFile(eafc) as z:
            for edition in [25,26,27]:
                f=f'fc{edition}_teams.csv'
                if f in z.namelist():
                    d=pd.read_csv(z.open(f),usecols=['teamid','teamname'],low_memory=False)
                    d=d.rename(columns={'teamid':'club_id','teamname':'club_name'});d['fifa_version']=edition
                    arr.append(d)
    if not arr:return pd.DataFrame(columns=['fifa_version','club_id','club_name','key'])
    x=pd.concat(arr,ignore_index=True).drop_duplicates(['fifa_version','club_id','club_name'])
    x['key']=x.club_name.map(norm)
    return x

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='output_final');ap.add_argument('--out',default='coverage_audit')
    ap.add_argument('--known-map',default='ucl_25_27_verified_mapping.csv')
    ap.add_argument('--fixtures-dir',default='fixtures');ap.add_argument('--offline',action='store_true');ap.add_argument('--roster-csv',default=None)
    ap.add_argument('--history-teams',default='male_teams.csv');ap.add_argument('--fc24-zip',default='24(1).zip');ap.add_argument('--eafc-zip',default='eafc_db_output.zip')
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    clubs=pd.read_csv(Path(a.input)/'clubs_2015_2027.csv',usecols=['season','fifa_version','club_id','club_name'],low_memory=False)
    clubs['key']=clubs.club_name.map(norm)
    catalog=combined_raw_team_catalog(Path(a.history_teams),Path(a.fc24_zip),Path(a.eafc_zip))
    known=pd.read_csv(a.known_map) if Path(a.known_map).exists() else pd.DataFrame()
    rows=[];counts=[]
    for season in sorted(clubs.season.unique()):
        version=int(clubs.loc[clubs.season==season,'fifa_version'].iloc[0])
        current=clubs[clubs.season==season];all_edition=catalog[catalog.fifa_version==version]
        if version<=24:
            cache=pd.read_csv(a.roster_csv) if a.roster_csv and Path(a.roster_csv).exists() else None
            try:names=sorted(cache.loc[cache.season==season,'uefa_team'].tolist()) if cache is not None else fixtures_for_season(season,Path(a.fixtures_dir),a.offline)
            except Exception as e:names=None;error=str(e)
            if names is None or len(names)!=32:
                counts.append({'season':season,'expected_teams':32,'teams_checked':0 if names is None else len(names),'matched_selected':None,'not_in_selected':None,'status':'NOT_CHECKED_SOURCE_UNAVAILABLE' if names is None else 'NOT_CHECKED_SOURCE_PARSE_FAILED','reason':error if names is None and 'error' in locals() else 'Require exactly 32 main-stage clubs; do not count qualifiers'})
                continue
        else:
            names=known.loc[known.season==season,'uefa_team'].dropna().tolist() if len(known) else []
            if len(names)!=36:
                counts.append({'season':season,'expected_teams':36,'teams_checked':len(names),'status':'NOT_CHECKED_MISSING_VERIFIED_MAP'});continue
        for name in names:
            key=norm(name);hits=current[current.key==key];status='UNRESOLVED';method='';cid=None;game_name=None;explanation=''
            verified=known[(known.season==season)&(known.uefa_team==name)] if len(known) else pd.DataFrame()
            # Audited club identity IDs, never fuzzy suggestions, and only if ID appears in current edition.
            if name in REVIEWED_CLUB_IDS and status=='UNRESOLVED':
                x=current[current.club_id==REVIEWED_CLUB_IDS[name]]
                if len(x)==1:status='MATCHED_SELECTED';method='reviewed_game_club_id';hits=x
            if len(verified):
                tid=pd.to_numeric(verified.iloc[0].team_id,errors='coerce')
                if pd.notna(tid):
                    hits=current[current.club_id.eq(int(tid))]
                    if len(hits)==1:status='MATCHED_SELECTED';method='reviewed_id'
                elif version>=25:status='NOT_FOUND_IN_AVAILABLE_GAME_EDITION';method='prior_review_no_id';explanation='Previously manually reviewed as missing in the game export'
            if status=='UNRESOLVED' and len(hits)==1:
                status='MATCHED_SELECTED';method='unique_exact_normalized_name_or_alias'
            if status=='MATCHED_SELECTED':
                cid=int(hits.iloc[0].club_id);game_name=str(hits.iloc[0].club_name)
            elif status=='UNRESOLVED':
                other=all_edition[all_edition.key==key]
                if len(other)==1:
                    status='FOUND_OTHER_GAME_UPDATE';cid=int(other.iloc[0].club_id);game_name=str(other.iloc[0].club_name)
                    explanation='Not in selected preseason snapshot; appears in another available snapshot'
                elif len(other)>1:
                    status='REVIEW_MULTIPLE_GAME_IDS';explanation='Multiple possible game teams share same normalized name'
                else:status='NOT_FOUND_IN_AVAILABLE_GAME_EDITION';explanation='Not found in available source snapshots (not proof absent from every EA edition)'
            scored=[]
            if status!='MATCHED_SELECTED':
                for d in current.itertuples():
                    sim=SequenceMatcher(None,key,d.key).ratio()
                    if sim>=0.55:scored.append((sim,d.club_name,int(d.club_id)))
                scored=sorted(scored,reverse=True)[:5]
            rows.append({'season':season,'fifa_version':version,'uefa_team':name,'club_id':cid,'game_team':game_name,'status':status,'match_method':method,'potential_candidates':' | '.join(f'{n} [ID {i}, similarity {s:.2f}]' for s,n,i in scored),'explanation':explanation,'source':BASE.format(season=season) if version<=24 else 'reviewed_2024_27_UEFA_game_map','sofifa_search_url':'https://sofifa.com/teams','ea_ratings_url':'https://www.ea.com/games/ea-sports-fc/ratings'})
        subset=[r for r in rows if r['season']==season]
        counts.append({'season':season,'expected_teams':32 if version<=24 else 36,'teams_checked':len(names),'matched_selected':sum(x['status']=='MATCHED_SELECTED' for x in subset),'not_in_selected':sum(x['status']!='MATCHED_SELECTED' for x in subset),'status':'CHECKED','reason':'Candidates require manual verification; NOT_FOUND is relative to available snapshots'})
    df=pd.DataFrame(rows);sm=pd.DataFrame(counts)
    df.to_csv(out/'ucl_teams_coverage.csv',index=False,encoding='utf-8-sig')
    df[df.status!='MATCHED_SELECTED'].to_csv(out/'ucl_unmatched_review.csv',index=False,encoding='utf-8-sig')
    sm.to_csv(out/'ucl_coverage_summary.csv',index=False,encoding='utf-8-sig')
    print(sm.to_string(index=False))
    if len(df):print('\nMissing/Review\n',df[df.status!='MATCHED_SELECTED'][['season','uefa_team','status','potential_candidates']].to_string(index=False))
    print('\nIMPORTANT: NOT_FOUND means absent from available snapshots, not definitive unlicensed by EA.')
if __name__=='__main__':main()
