from pathlib import Path
import tempfile
import unittest
import pandas as pd
import numpy as np
from match_builder.features import select_xi, POS_ROLE
from match_builder.mapping import make_mapping
from build_match_dataset import apply_90min_corrections, create_labels


class MatchBuilderTests(unittest.TestCase):
    def test_defender_shortage_does_not_shift_midfield_slots(self):
        pos=['GK','CB','CB','CB','CM','CM','CM','ST','ST','ST','ST']
        team=pd.DataFrame({'player_id':range(1,12),'overall':list(range(90,79,-1)),
                           'player_positions':pos,'club_position':pos})
        team['__role']=team.player_positions.map(POS_ROLE)
        chosen=select_xi(team)
        self.assertIsNotNone(chosen)
        xi, provenance=chosen
        self.assertEqual(len(xi),11)
        self.assertEqual(provenance[4],'fill')
        self.assertEqual([p['__role'] for p in xi[5:8]],['mid']*3)
        self.assertEqual(len({p['player_id'] for p in xi}),11)

    def test_verified_inter_alias_and_absent_club(self):
        m=pd.DataFrame([{'season':'2025-26','home_team':'FC Internazionale Milano','away_team':'Paphos FC'}])
        c=pd.DataFrame([{'season':'2025-26','club_id':131682,'club_name':'Lombardia FC'}])
        coverage=pd.DataFrame(columns=['season','uefa_team','club_id'])
        verified=pd.DataFrame([{'season':'2025-26','uefa_team':'Inter','team_id':131682,'status':'alias_confirmed'},
                               {'season':'2025-26','uefa_team':'Pafos','team_id':np.nan,'status':'not_in_game'}])
        map_=make_mapping(m,c,coverage,None,verified).set_index('match_team')
        self.assertEqual(int(map_.loc['FC Internazionale Milano','club_id']),131682)
        self.assertEqual(map_.loc['Paphos FC','status'],'UNMATCHED')

    def test_corrected_extra_time_label(self):
        sample=pd.DataFrame([{'season':'2015-16','match_date':'2016-03-16',
              'home_team':'Bayern Munich','away_team':'Juventus','extra_time':True,
              'status':'played','home_goals_raw':4,'away_goals_raw':2,
              'home_goals_90':np.nan,'away_goals_90':np.nan}])
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'correct.csv'
            pd.DataFrame([{'season':'2015-16','match_date':'2016-03-16',
             'home_team':'Bayern Munich','away_team':'Juventus',
             'home_goals_90':2,'away_goals_90':2,
             'home_goals_final_expected':4,'away_goals_final_expected':2}]).to_csv(p,index=False)
            corrected=create_labels(apply_90min_corrections(sample,p))
        self.assertEqual(corrected.loc[0,'label_90min_wdl'],'D')
        self.assertEqual(corrected.loc[0,'label_90min_goal_diff'],0)

    def test_scheduled_match_unlabeled(self):
        x=pd.DataFrame([{'status':'scheduled','home_goals_90':np.nan,'away_goals_90':np.nan}])
        result=create_labels(x)
        self.assertTrue(pd.isna(result.loc[0,'label_90min_wdl']))


if __name__=='__main__':
    unittest.main()
