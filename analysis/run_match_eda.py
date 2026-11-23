"""Reproducible UCL match EDA. Run from repository root:
python analysis/run_match_eda.py --data data/match_dataset --out analysis/eda_output
No training data files are changed. Analysis-only strength differences are NOT model features.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def export(fig, out, name):
    fig.tight_layout(); fig.savefig(out / f'{name}.png',dpi=180,bbox_inches='tight'); fig.savefig(out / f'{name}.pdf',bbox_inches='tight'); plt.close(fig)


def run(data,out):
    out=Path(out); figs=out/'figures'; tables=out/'tables'; figs.mkdir(parents=True,exist_ok=True);tables.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(Path(data)/'matches_compact.csv',low_memory=False)
    full=pd.read_csv(Path(data)/'matches_full.csv',low_memory=False)
    status=pd.read_csv(Path(data)/'sample_status.csv',low_memory=False)
    assert len(d)==len(full) and set(d.match_id)==set(full.match_id)
    goal_h='label_90min_home_goals';goal_a='label_90min_away_goals'; y='label_90min_wdl'
    for name,df in [('compact',d),('full',full)]:
        # Features only; metadata, post-match flags and labels excluded.
        with open(Path(data)/f'{name}_feature_columns.json',encoding='utf-8') as f: schema=json.load(f)
        cols=schema if isinstance(schema,list) else next(v for v in schema.values() if isinstance(v,list))
        cols=[c for c in cols if c in df.columns]
        miss=df[cols].isna().mean().sort_values(ascending=False)
        missing=pd.DataFrame({'feature':miss.index,'missing_rate':miss.values,'missing_count':[int(df[c].isna().sum()) for c in miss.index], 'unique_nonnull':[df[c].nunique(dropna=True) for c in miss.index]})
        missing['recommendation']=np.select([missing.missing_rate.eq(0)&missing.unique_nonnull.le(1),missing.missing_rate.ge(.9),missing.feature.str.contains('bench_att_best_overall'),missing.feature.str.contains('mentality_composure'),missing.missing_rate.gt(0)],['drop_constant','exclude_unavailable','missing_indicator_plus_train_median','train_median_with_version_warning','train_median'],default='keep')
        missing.to_csv(tables/f'{name}_missingness.csv',index=False)
        if name=='full':
            for c in ['home_att1_mentality_composure','away_att1_mentality_composure']:
                if c in df:
                    df.groupby('season')[c].apply(lambda s:s.isna().mean()).rename('missing_rate').to_csv(tables/f'{c}_by_season.csv')
    counts=status.groupby('season').agg(total=('match_id','size'),eligible=('eligible_historical',lambda s:sum(s.astype(str).str.lower()=='true'))).reset_index()
    counts['coverage']=counts['eligible']/counts['total'];counts.to_csv(tables/'coverage_by_season.csv',index=False)
    m=d.copy();m['goal_diff']=m[goal_h]-m[goal_a];m['total_goals']=m[goal_h]+m[goal_a]
    m['rating_gap']=m.home_club_overall-m.away_club_overall
    m['xi_gap']=m.home_xi_overall_mean-m.away_xi_overall_mean
    m['bench_gap']=m.home_bench_overall_mean-m.away_bench_overall_mean
    m['home_points']=m[y].map({'H':3,'D':1,'A':0})
    m['home_win']=(m[y]=='H').astype(int);m['away_win']=(m[y]=='A').astype(int);m['draw']=(m[y]=='D').astype(int)
    rates=m.groupby('season').agg(matches=('match_id','size'),home_win=('home_win','mean'),draw=('draw','mean'),away_win=('away_win','mean'),home_goals=(goal_h,'mean'),away_goals=(goal_a,'mean'),mean_goals=('total_goals','mean')).reset_index();rates.to_csv(tables/'season_rates.csv',index=False)
    # 01 coverage
    fig,ax=plt.subplots(figsize=(11,4));ax.bar(counts.season,counts.eligible,label='Eligible');ax.bar(counts.season,counts.total-counts.eligible,bottom=counts.eligible,label='Excluded');ax.tick_params(axis='x',rotation=45);ax.set_ylabel('Matches');ax.set_title('Historical match coverage by season');ax.legend();export(fig,figs,'01_coverage')
    # 02 outcomes
    fig,ax=plt.subplots(figsize=(6,4));v=m[y].value_counts().reindex(['H','D','A']);ax.bar(['Home win','Draw','Away win'],v.values);ax.set_ylabel('Matches');ax.set_title('90-minute outcomes (eligible sample)');[ax.text(i,val+8,f'{val} ({val/len(m):.1%})',ha='center') for i,val in enumerate(v)];export(fig,figs,'02_outcomes')
    # 03 annual home goals split
    fig,ax=plt.subplots(figsize=(10,4));ax.plot(rates.season,rates.home_goals,marker='o',label='Home goals');ax.plot(rates.season,rates.away_goals,marker='o',label='Away goals');ax.tick_params(axis='x',rotation=45);ax.set_ylabel('Goals / match');ax.set_title('Home and away scoring by season');ax.legend();export(fig,figs,'03_goals_by_season')
    # 04 league vs knockout
    bystage=m.groupby('stage').agg(n=('match_id','size'),home_win=('home_win','mean'),draw=('draw','mean'),away_win=('away_win','mean'),goals=('total_goals','mean')).reset_index();bystage.to_csv(tables/'stage_outcomes.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4));left=np.zeros(len(bystage));
    for k,lab in [('home_win','Home'),('draw','Draw'),('away_win','Away')]:ax.bar(bystage.stage,bystage[k],bottom=left,label=lab);left+=bystage[k].to_numpy()
    ax.set_ylim(0,1);ax.set_ylabel('Share');ax.set_title('Outcome mix by competition stage');ax.legend();export(fig,figs,'04_stage_outcomes')
    # 05 rating bins including sample counts
    edges=[-np.inf,-10,-5,-2,2,5,10,np.inf];labels=['<=-10','-10 to -5','-5 to -2','-2 to +2','+2 to +5','+5 to +10','>=+10']
    m['rating_gap_bin']=pd.cut(m.rating_gap,bins=edges,labels=labels,include_lowest=True)
    bins=m.groupby('rating_gap_bin',observed=False).agg(n=('match_id','size'),home_win=('home_win','mean'),draw=('draw','mean'),away_win=('away_win','mean'),goal_diff=('goal_diff','mean')).reset_index();bins.to_csv(tables/'rating_gap_bins.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,4));ax.plot(bins.rating_gap_bin.astype(str),bins.home_win,marker='o',label='Home win');ax.plot(bins.rating_gap_bin.astype(str),bins.draw,marker='o',label='Draw');ax.plot(bins.rating_gap_bin.astype(str),bins.away_win,marker='o',label='Away win');
    for i,r in bins.iterrows():ax.annotate(f'n={int(r.n)}',(i,min(0.98,r.home_win+.055)),ha='center',fontsize=8)
    ax.set_ylim(0,1.12);ax.set_xlabel('Home - away club overall');ax.set_ylabel('Observed probability');ax.legend();ax.set_title('Result versus relative club strength');export(fig,figs,'05_rating_gap_results')
    # 06 within evenly matched games: home edge (descriptive)
    near=m.loc[m.rating_gap.abs()<=2];
    # 07 individual and aggregate rating associations
    corrs={}
    for v in ['rating_gap','xi_gap','bench_gap']:
        corrs[v]={'pearson_goal_diff':float(m[[v,'goal_diff']].corr().iloc[0,1]),'n':int(m[v].notna().sum())}
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(list(corrs),[x['pearson_goal_diff'] for x in corrs.values()]);ax.set_ylim(-.1,1);ax.set_ylabel('Pearson correlation with 90-min goal difference');ax.set_title('Strength measures: descriptive correlations');export(fig,figs,'06_strength_correlation')
    # 08 scoring distribution: Poisson motivation
    fig,ax=plt.subplots(figsize=(8,4));maxg=int(min(9,m.total_goals.max()));g=m.total_goals.clip(upper=maxg).value_counts().reindex(range(maxg+1),fill_value=0);ax.bar(g.index,g.values);ax.set_xticks(range(maxg+1));ax.set_xticklabels([str(k) for k in range(maxg)]+[f'{maxg}+']);ax.set_xlabel('90-minute total goals');ax.set_ylabel('Matches');ax.set_title('Total-goal distribution');export(fig,figs,'07_total_goals')
    # 09 missingness top 30 for full
    miss=pd.read_csv(tables/'full_missingness.csv');mm=miss.query('missing_rate > 0').head(30).iloc[::-1]
    fig,ax=plt.subplots(figsize=(10,8));ax.barh(mm.feature,mm.missing_rate*100);ax.set_xlabel('Missing (%)');ax.set_title('Full dataset: top missing predictors');ax.tick_params(axis='y',labelsize=7);export(fig,figs,'08_missingness')
    # 10 stratified rating gap across stage
    stage_bins=m.groupby(['stage','rating_gap_bin'],observed=False).agg(n=('match_id','size'),home_win=('home_win','mean')).reset_index();stage_bins.to_csv(tables/'rating_gap_by_stage.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,4));
    for stage,grp in stage_bins.groupby('stage'):
        ax.plot(grp.rating_gap_bin.astype(str),grp.home_win,marker='o',label=stage)
    ax.set_ylim(0,1);ax.legend();ax.set_xlabel('Home - away club overall');ax.set_ylabel('Home win rate');ax.set_title('Strength gap relationship across stages');export(fig,figs,'09_strength_by_stage')
    summary={'matches':len(m),'home_wins':int(m.home_win.sum()),'draws':int(m.draw.sum()),'away_wins':int(m.away_win.sum()),'mean_home_goals':float(m[goal_h].mean()),'mean_away_goals':float(m[goal_a].mean()),'mean_total_goals':float(m.total_goals.mean()),'mean_goal_diff':float(m.goal_diff.mean()),'near_equal_n':len(near),'near_equal_home_win':float(near.home_win.mean()),'near_equal_away_win':float(near.away_win.mean()),'near_equal_draw':float(near.draw.mean()),'correlations':corrs,'by_stage':bystage.to_dict(orient='records'),'missing_compact_nonzero':pd.read_csv(tables/'compact_missingness.csv').query('missing_rate>0').to_dict(orient='records'),'missing_full_nonzero_count':int((pd.read_csv(tables/'full_missingness.csv').missing_rate>0).sum())}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/match_dataset');p.add_argument('--out',default='analysis/eda_output');a=p.parse_args();run(a.data,a.out)
