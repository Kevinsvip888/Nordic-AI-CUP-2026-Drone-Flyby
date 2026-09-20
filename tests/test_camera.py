import numpy as np
import lifecycle_helpers as lifecycle
from camera_policy import MemoryCameraPolicy

def groups(a):return MemoryCameraPolicy(lifecycle.Motion()).groups(a)
def test_regression():
    lifecycle.MemoryCameraPolicy=MemoryCameraPolicy;lifecycle.run()
    p=MemoryCameraPolicy(lifecycle.Motion())
    motion={'accepted':True,'matrix':[[1,0,0],[0,1,0]],'dt':1,'residual_px':.2}
    previous=groups([lifecycle.det((300,200,320,220),.2,'tank'),lifecycle.det((650,250,670,270),.2,'hangar')])
    changed=groups([lifecycle.det((300,200,320,220),.21,'condor'),lifecycle.det((650,250,670,270),.49,'hangar')])
    for i in range(2):
        cmd,_,_=p.camera(i,0,previous,{},[],motion,[]);assert cmd is None
    cmd,target,_=p.camera(2,0,changed,{},[],motion,[])
    assert cmd['resolution_level']==1 and target['change']['class_changed'] and target['object_id']=='condor'
    assert target['l0_observations_before_zoom']==3 # class change beats larger score delta
    cmd,_,_=p.camera(3,1,changed,{},[],motion,[]);assert cmd=={'resolution_level':0,'center_x':1920,'center_y':1080}
    for i in range(4,6):
        cmd,_,_=p.camera(i,0,previous if i%2 else changed,{},[],motion,[]);assert cmd is None
    p=MemoryCameraPolicy(lifecycle.Motion());p.l0_streak=2;p.previous_groups=previous
    only_scores=groups([lifecycle.det((300,200,320,220),.3,'tank'),lifecycle.det((650,250,670,270),.45,'hangar')])
    cmd,target,_=p.camera(2,0,only_scores,{},[],motion,[])
    assert cmd['resolution_level']==1 and target['object_id']=='hangar' and not target['change']['class_changed']
    print('PASS: three L0 views, class-change priority over score delta, largest confidence delta fallback, mandatory L1 return')
if __name__=='__main__':test_regression()

import unittest

class PolicyRegression(unittest.TestCase):
    def test_policy(self):
        test_regression()
