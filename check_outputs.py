from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parent/'output'
required={'players_2015_2027.csv':['season','player_id','club_id','overall','potential','attacking_finishing'], 'clubs_2015_2027.csv':['season','club_id','overall','attack','midfield','defence'], 'coaches_2015_2027.csv':['season','club_id','coach_id'], 'club_features_2015_2027.csv':['season','club_id','overall_top11_mean']}
for f,fields in required.items():
 path=root/f
 if not path.exists():raise FileNotFoundError(path)
 d=pd.read_csv(path,usecols=list(set(fields)),low_memory=False)
 assert all(x in d.columns for x in fields)
 if f.startswith('players'):assert not d.duplicated(['season','club_id','player_id']).any()
 else:assert not d.duplicated(['season','club_id']).any()
 print(f, len(d), 'rows;', d.season.nunique(),'seasons')
print('Output checks passed.')
