"""Audit SoFIFA FC24 ZIP and FIFA15–23 CSVs without loading full players CSV into RAM.

Usage:
  python inspect_sofifa.py --fc24 "24.zip" --history-dir "./fifa15_23"
  python inspect_sofifa.py --fc24 "24.zip" --history-dir "." --out "./audit"

Requires: pip install pandas
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import zipfile
import pandas as pd

CHUNK = 100_000
FILES = ('male_players.csv','male_teams.csv','male_coaches.csv')
HISTORY = {
    'players_full': 'male_players.csv',
    'players_legacy': 'male_players (legacy).csv',
    'teams': 'male_teams.csv',
    'coaches': 'male_coaches.csv',
}
KEYS = [
    'fifa_version','fifa_update','fifa_update_date','update_as_of',
    'player_id','sofifa_id','club_team_id','team_id','coach_id',
    'club_name','team_name','overall','potential',
]

def inspect_csv(source, label, out_dir):
    """source is a path, or an already opened binary zip member."""
    columns = pd.read_csv(source, nrows=0).columns.tolist()
    if hasattr(source, 'seek'):
        source.seek(0)
    wanted = [c for c in KEYS if c in columns]
    counts_by_year = Counter()
    counts_by_update = Counter()
    missing = Counter()
    n = 0
    examples = []
    for part in pd.read_csv(source, usecols=wanted or None,
                            chunksize=CHUNK, low_memory=False):
        n += len(part)
        if not examples:
            examples = part.head(3).to_dict(orient='records')
        for c in wanted:
            missing[c] += int(part[c].isna().sum())
        if 'fifa_version' in part:
            counts_by_year.update(part['fifa_version'].fillna('NA').astype(str).value_counts().to_dict())
            date_col = next((c for c in ('fifa_update_date','update_as_of') if c in part), None)
            grp = ['fifa_version']
            if 'fifa_update' in part:
                grp.append('fifa_update')
            if date_col:
                grp.append(date_col)
            batch = part.groupby(grp, dropna=False).size()
            counts_by_update.update({tuple(map(str,k if isinstance(k,tuple) else (k,))): int(v) for k,v in batch.items()})
    prefix = label.replace(' ','_').replace('/','_')
    pd.DataFrame({'column': columns}).to_csv(out_dir / f'{prefix}_columns.csv', index=False)
    pd.DataFrame([{'version':k,'rows':v} for k,v in sorted(counts_by_year.items())]).to_csv(
        out_dir / f'{prefix}_versions.csv', index=False)
    pd.DataFrame([{'group': ' | '.join(k), 'rows':v} for k,v in sorted(counts_by_update.items())]).to_csv(
        out_dir / f'{prefix}_updates.csv', index=False)
    with (out_dir / f'{prefix}_summary.txt').open('w',encoding='utf-8') as f:
        f.write(f'{label}\nRows: {n:,}\nColumns: {len(columns)}\n\n')
        f.write('ALL COLUMNS:\n'+'\n'.join(columns)+'\n\n')
        f.write('BY VERSION:\n'+'\n'.join(f'  {k}: {v:,}' for k,v in sorted(counts_by_year.items()))+'\n\n')
        f.write('KEY COLUMN MISSING RATES:\n')
        f.writelines(f'  {c}: {missing[c]/n:.1%}\n' for c in wanted if n)
        f.write(f'\nNUMBER OF VERSION / UPDATE / DATE COMBINATIONS: {len(counts_by_update)}\n')
        f.write('\nFIRST 3 ROWS (KEY COLUMNS):\n'+pd.DataFrame(examples).to_string(index=False)+'\n')
    print(f'  {label:25} rows={n:>10,}   cols={len(columns):>3}   updates={len(counts_by_update):>5}')
    return set(columns)

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--fc24',type=Path,default=Path('24.zip'))
    ap.add_argument('--history-dir',type=Path,default=Path('.'))
    ap.add_argument('--out',type=Path,default=Path('sofifa_audit'))
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    schemas={}
    print('=== FC24 zip ===')
    if args.fc24.exists():
        with zipfile.ZipFile(args.fc24) as z:
            members={Path(name).name:name for name in z.namelist()}
            for fname in FILES:
                if fname not in members:
                    print('  MISSING:',fname)
                    continue
                with z.open(members[fname]) as f:
                    label='fc24_'+fname.removesuffix('.csv')
                    schemas[label]=inspect_csv(f,label,args.out)
    else:
        print('  ZIP NOT FOUND:',args.fc24)
    print('\n=== Historical FIFA 15–23 ===')
    for key, fname in HISTORY.items():
        p=args.history_dir/fname
        label='history_'+key
        if p.exists():
            schemas[label]=inspect_csv(p,label,args.out)
        else:
            print('  NOT FOUND:',p)
    print('\n=== Column comparisons ===')
    pairs=[('fc24_male_players','history_players_full'),
           ('fc24_male_players','history_players_legacy'),
           ('history_players_full','history_players_legacy'),
           ('fc24_male_teams','history_teams'),
           ('fc24_male_coaches','history_coaches')]
    with (args.out/'column_comparison.txt').open('w',encoding='utf-8') as f:
        for a,b in pairs:
            if a not in schemas or b not in schemas: continue
            sa,sb=schemas[a],schemas[b]
            msg=(f'\n{a} vs {b}: {len(sa & sb)} shared\n'
                 f'  Only {a}: {sorted(sa-sb)}\n'
                 f'  Only {b}: {sorted(sb-sa)}\n')
            print(msg,end='')
            f.write(msg)
    print('\nSaved audit reports to:',args.out.resolve())
    print('Important: update date labels alone do not establish the actual publication time.')

if __name__=='__main__':
    main()
