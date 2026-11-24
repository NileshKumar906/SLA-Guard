import pandas as pd
from sla_guard.features import norm_target,group_incidents,make_X
def test_target(): assert norm_target(pd.Series(["true","false","1","0"])).tolist()==[1,0,1,0]
def test_leak_exclusion():
 d=pd.DataFrame({"sys_id":["a","a","b"],"made_sla":[True,True,False],"opened_at":["2024-01-01","2024-01-02","2024-01-03"],"priority":["2","2","1"],"close_code":["x","x","y"],"sys_mod_count":[4,5,3]}); inc,t,i,o=group_incidents(d); X,y,_=make_X(inc,t,i,o); assert len(inc)==2 and "close_code" not in X and "sys_mod_count" not in X
