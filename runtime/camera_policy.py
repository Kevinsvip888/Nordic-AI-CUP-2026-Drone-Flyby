"""Same verified Kalman tracking, slower camera, observed class changes first."""
from copy import deepcopy
import numpy as np
from scipy.optimize import linear_sum_assignment
from tracking import MemoryCameraPolicy as BasePolicy,center,sides,warp,iou,inside,FRAME,SIZE,EXCLUDED

class MemoryCameraPolicy(BasePolicy):
    def __init__(self,motion=None):
        super().__init__(motion);self.previous_groups=[]
    def configuration(self):
        return {**super().configuration(),'camera':'L0-L1-L0; at least three received consecutive L0 views before inspection; no L2',
                'camera_priority':'uniquely associated actual class change first; then absolute top-class confidence change',
                'comparison_source':'successive real observed detector groups, never propagated identity scores'}
    def changes(self,groups,motion):
        previous=self.previous_groups;self.previous_groups=deepcopy(groups)
        if not previous or not motion['accepted']:return
        costs=np.full((len(previous),len(groups)),1e6)
        for pi,p in enumerate(previous):
            predicted=warp(p['box'],motion['matrix'])
            for gi,g in enumerate(groups):
                if min(p['score'],g['score'])<.1:continue
                ratio=float(np.max(np.maximum(sides(g['box'])/sides(predicted),sides(predicted)/sides(g['box']))))
                distance=float(np.linalg.norm(center(predicted)-center(g['box'])));overlap=iou(predicted,g['box'])
                gate=min(18.,max(4.,.6*np.linalg.norm(sides(predicted))))
                if ratio>1.8 or distance>gate or (overlap<.15 and distance>max(3.,.5*min(sides(predicted)))):continue
                costs[pi,gi]=.6*distance/gate+.4*(1-overlap)
        if not costs.size:return
        for pi,gi in zip(*linear_sum_assignment(costs)):
            score=costs[pi,gi];row=np.sort(costs[pi]);col=np.sort(costs[:,gi])
            if score>=1e5 or score>row[0]+1e-9 or score>col[0]+1e-9:continue
            if (len(row)>1 and row[1]-row[0]<.12) or (len(col)>1 and col[1]-col[0]<.12):continue
            p=previous[pi];g=groups[gi]
            g['change']={'class_changed':p['label']!=g['label'],'previous_class':p['label'],
                         'current_class':g['label'],'previous_confidence':p['score'],'current_confidence':g['score'],
                         'confidence_delta':abs(g['score']-p['score'])}
    def camera(self,index,level,groups,matches,tracks,motion,events):
        self.changes(groups,motion)
        if motion['accepted']:
            for old in self.cooldowns:old['box']=warp(old['box'],motion['matrix'])
        self.cooldowns=[c for c in self.cooldowns if index-c['index']<=5]
        if level==1:
            self.l0_streak=0;return {'resolution_level':0,'center_x':1920,'center_y':1080},None,'return_L0'
        self.l0_streak+=1
        if self.l0_streak<3:return None,None,'hold_L0_minimum_three_observations'
        if not motion['accepted']:return None,None,'hold_L0_motion_unreliable'
        if any(t.seen!=index for t in tracks):return None,None,'hold_L0_unresolved_identity'
        dt=motion['dt'];mat=np.asarray(motion['matrix']);nextmat=np.c_[np.eye(2)+(mat[:,:2]-np.eye(2))/dt,mat[:,2]/dt]
        forecasts={}
        for t in tracks:
            k=deepcopy(t.kalman);k.predict(1,nextmat,True);forecasts[t.id]=k
        def future(t):
            wh=sides(warp(t.box,nextmat));c=forecasts[t.id].x[:2];return np.r_[c-wh/2,c+wh/2]
        choices=[]
        for g in groups:
            change=g.get('change')
            if not change or (not change['class_changed'] and change['confidence_delta']<=1e-6):continue
            b=g['box'];w,h=sides(b)
            if min(w,h)<2 or max(w,h)>100:continue
            pred=warp(b,nextmat)
            target_track=next((t for t in tracks if t.seen==index and iou(b,t.observed_box)>=.5),None)
            if target_track is not None:pred=future(target_track)
            if not inside(pred,FRAME,3):continue
            if any(np.linalg.norm(center(pred)-center(c['box']))<35 for c in self.cooldowns):continue
            c=center(pred);cx=4*round(float(np.clip(c[0],240,720)));cy=4*round(float(np.clip(c[1],135,405)))
            crop=np.array([cx/4-240,cy/4-135,cx/4+240,cy/4+135])
            if any(not t.persistent and index+1-t.confirmed>=3 and not inside(future(t),crop) for t in tracks):continue
            if any(not inside(future(t),crop) and 2*forecasts[t.id].position_sigma()+max(.5,motion['residual_px'])>max(3.,.28*min(sides(future(t)))) for t in tracks):continue
            choices.append(((int(change['class_changed']),change['confidence_delta']),g,pred,cx,cy))
        if not choices:return None,None,'hold_L0_no_safe_observed_change'
        rank,g,pred,cx,cy=max(choices,key=lambda x:x[0]);self.cooldowns.append({'index':index,'box':pred.copy()});self.last_zoom=index
        return {'resolution_level':1,'center_x':int(cx),'center_y':int(cy)}, {
            'object_id':g['label'],'confidence':g['score'],'priority':[rank[0],float(rank[1])],
            'change':g['change'],'l0_observations_before_zoom':self.l0_streak,
            'predicted_bbox':(pred/SIZE).tolist()},'inspect_L1'
