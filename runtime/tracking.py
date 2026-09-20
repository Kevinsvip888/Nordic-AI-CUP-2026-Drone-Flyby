"""Observation-corrected identities with bounded, visibility-aware position outputs."""
from copy import deepcopy
from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from visible_view_motion import VisibleViewMotion
from kalman_position import KalmanPosition

EXCLUDED={'ta-ta','small_launcher'}
SIZE=np.array([960.,540.,960.,540.])
FRAME=np.array([0.,0.,960.,540.])
def area(b):return max(0.,b[2]-b[0])*max(0.,b[3]-b[1])
def center(b):return np.array([(b[0]+b[2])/2,(b[1]+b[3])/2])
def sides(b):return np.maximum(.01,np.asarray(b[2:])-b[:2])
def clip(b):return np.maximum(FRAME[:2].tolist()*2,np.minimum(b,FRAME[2:].tolist()*2))
def intersection(a,b):return np.array([max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])])
def iou(a,b):
    hit=area(intersection(a,b));return hit/max(1e-9,area(a)+area(b)-hit)
def inside(b,r,margin=0):return r[0]+margin<=b[0] and r[1]+margin<=b[1] and b[2]<=r[2]-margin and b[3]<=r[3]-margin
def warp(b,m):
    pts=np.array([[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]])@np.asarray(m)[:,:2].T+np.asarray(m)[:,2]
    return np.r_[pts.min(axis=0),pts.max(axis=0)]

@dataclass
class Track:
    id:int
    label:str
    box:np.ndarray
    score:float
    confirmed:int
    seen:int
    pose_index:int
    persistent:bool
    kalman:KalmanPosition
    observed_box:np.ndarray | None=None
    visible_misses:int=0
    uncertainty:float=1.
    position_valid:bool=True
    label_candidate:str=''
    label_hits:int=0

class MemoryCameraPolicy:
    def __init__(self,motion=None):
        self.motion_engine=motion or VisibleViewMotion();self.tracks=[];self.last=None;self.next_id=1
        self.l0_streak=0;self.last_zoom=-10;self.cooldowns=[]

    def configuration(self):
        return {'seed_confidence_gt':.3,'persistent_confidence_gt':.5,'medium_confirmation_age':3,
         'excluded_classes':sorted(EXCLUDED),'position':'per-track Kalman center/residual-velocity plus global visible-overlap motion; update on each reliable observation; display observed boxes without lag',
         'kalman_state':'source-plane cx,cy,vx,vy; camera affine control; covariance-aware matching and memory output',
         'memory_output_max_unseen_frames':2,'visible_unmatched_output':False,'visible_misses_to_retire':2,
         'edge_unmatched_to_retire':1,'identity_without_position':'retained but not emitted; camera returns to L0',
         'camera':'L0-L1-L0, two L0 observations before another inspection','no_hidden_pixels':True}

    def groups(self,raw):
        groups=[]
        for a in sorted(raw,key=lambda x:-x['confidence']):
            if a['object_id'] in EXCLUDED or a['confidence']<.05:continue
            b=np.asarray(a['bbox'])*SIZE
            g=next((g for g in groups if iou(b,g['box'])>=.65),None)
            if g is None:
                g={'box':b,'label':a['object_id'],'score':a['confidence'],'scores':{},'raw':[]};groups.append(g)
            g['scores'][a['object_id']]=max(g['scores'].get(a['object_id'],0),a['confidence']);g['raw'].append(a)
        return groups

    def step(self,index,level,region,raw,observed_bgr):
        assert self.last is None or index>self.last
        motion=self.motion_engine.estimate(index,region,observed_bgr);dt=index-self.last if self.last is not None else 1
        reliable=motion['accepted'];matrix=motion['matrix'];region=np.asarray(region)/4
        groups=self.groups(raw);events=[];predictions=[]
        for t in self.tracks:
            predicted_center=t.kalman.predict(dt,matrix,reliable)
            if reliable and t.position_valid:
                b=warp(t.box,matrix)
                wh=sides(b);b=np.r_[predicted_center-wh/2,predicted_center+wh/2]
                t.uncertainty=2*t.kalman.position_sigma()+max(.5,motion['residual_px'])
            else:b=t.box.copy();t.uncertainty+=4*dt
            predictions.append(b)
        costs=np.full((len(self.tracks),len(groups)),1e6);geometry={}
        for ti,t in enumerate(self.tracks):
            b=predictions[ti]
            for gi,g in enumerate(groups):
                box=g['box'];ratio=float(np.max(np.maximum(sides(box)/sides(b),sides(b)/sides(box))))
                overlap=iou(b,box);distance=float(np.linalg.norm(center(b)-center(box)))
                gate=min(24.,max(4.,.65*np.linalg.norm(sides(b)))+min(6.,2*t.kalman.position_sigma()))
                if not reliable:gate=min(40.,gate+12*min(dt,2))
                same=g['scores'].get(t.label,0.)>=.05
                if ratio>2. or distance>gate:continue
                if not same and (overlap<.45 or distance>.3*gate):continue
                if g['score']<.15 and (not same or overlap<.4 or distance>.4*gate):continue
                # Dormant identities require especially strong, current evidence to reactivate.
                if not t.position_valid and (not same or overlap<.5 or g['score']<=.3):continue
                cost=.55*distance/gate+.35*(1-overlap)+.1*abs(math.log(ratio))+(0 if same else .2)
                costs[ti,gi]=cost;geometry[ti,gi]={'distance_px':distance,'iou':overlap,'size_ratio':ratio,'same_class_candidate':same}
        matches={};ambiguous=set()
        if costs.size:
            for ti,gi in zip(*linear_sum_assignment(costs)):
                cost=costs[ti,gi]
                if cost>=1e5:continue
                row=np.sort(costs[ti]);col=np.sort(costs[:,gi])
                if cost>row[0]+1e-9 or cost>col[0]+1e-9:
                    ambiguous.add(ti);continue
                if (len(row)>1 and row[1]-row[0]<.12) or (len(col)>1 and col[1]-col[0]<.12):
                    ambiguous.add(ti);continue
                matches[ti]=gi
        output=[];kept=[];claimed=set();association=[]
        for ti,t in enumerate(self.tracks):
            pred=predictions[ti];matched=ti in matches;g=groups[matches[ti]] if matched else None
            if matched:
                gi=matches[ti];claimed.add(gi);association.append({'track_id':t.id,**geometry[ti,gi]})
                seen_gap=index-t.seen
                kupdate=t.kalman.update(center(g['box']),g['score']);association[-1]['kalman']=kupdate
                if not reliable:
                    # Without camera compensation, apparent speed is not identifiable
                    # as object motion. Keep measured position, reset residual speed.
                    t.kalman.x[2:]=0.;t.kalman.P[:2,2:]=0.;t.kalman.P[2:,:2]=0.;t.kalman.P[2:,2:]=np.eye(2)*4
                    association[-1]['kalman']['velocity_reset_without_camera_motion']=True
                # Current observation is the geometry authority; do not lag it with smoothing.
                wh=sides(g['box']);c=t.kalman.x[:2]
                t.box=np.r_[c-wh/2,c+wh/2];t.observed_box=g['box'].copy()
                t.seen=index;t.pose_index=index;t.visible_misses=0;t.uncertainty=2*t.kalman.position_sigma();t.position_valid=True
                class_score=g['scores'].get(t.label,0.)
                if class_score>.3 and (index-t.confirmed>=3 or class_score>.5):
                    t.confirmed=index;t.score=class_score;t.persistent=t.persistent or class_score>.5
                if g['label']!=t.label and g['score']>.5 and geometry[ti,gi]['iou']>=.45:
                    t.label_hits=t.label_hits+1 if t.label_candidate==g['label'] else 1;t.label_candidate=g['label']
                    if t.label_hits>=2:
                        t.label=g['label'];t.score=g['score'];t.confirmed=index;t.persistent=True
                        events.append({'event':'class_reconfirmed_twice','track_id':t.id});t.label_hits=0
                else:t.label_hits=0;t.label_candidate=''
                events.append({'event':'observation_position_update','track_id':t.id,'seen_gap':seen_gap})
                events.append({'event':'kalman_measurement_update','track_id':t.id})
                kind='observed_corrected'
            else:
                t.box=pred;t.pose_index=index
                if reliable and t.position_valid and area(clip(pred))<=0:
                    events.append({'event':'left_frame','track_id':t.id});continue
                visible=area(intersection(pred,region))>=.5*max(1e-6,area(clip(pred)))
                near_edge=not inside(pred,FRAME,3)
                if visible:
                    t.visible_misses+=1
                    if near_edge or t.visible_misses>=2:
                        events.append({'event':'edge_disappeared' if near_edge else 'visible_missing_retired','track_id':t.id});continue
                    events.append({'event':'visible_missing_hidden','track_id':t.id})
                if not reliable:t.position_valid=False
                kind='memory_prediction'
            if not t.persistent and index-t.confirmed>=3:
                if matched:claimed.discard(matches[ti])
                events.append({'event':'confirmation_expired','track_id':t.id});continue
            if area(clip(t.box))<=0:
                events.append({'event':'outside_hidden','track_id':t.id});continue
            kept.append(t)
            if not matched:
                visible=area(intersection(t.box,region))>=.5*max(1e-6,area(clip(t.box)))
                if visible or ti in ambiguous or not t.position_valid or not reliable or index-t.seen>2:
                    events.append({'event':'unsupported_position_hidden','track_id':t.id});continue
                if t.uncertainty>max(3.,.28*min(sides(t.box))):
                    events.append({'event':'uncertain_position_hidden','track_id':t.id});continue
            output.append(self.annotation(t,index,kind))
        # New identities are seeded only by real observations, never by propagated boxes.
        for gi,g in enumerate(groups):
            if gi in claimed or g['score']<=.3:continue
            # If geometry is ambiguous, do not attach identity; raw detection can still be shown.
            if any(iou(g['box'],t.box)>=.3 for t in kept):continue
            t=Track(self.next_id,g['label'],g['box'].copy(),g['score'],index,index,index,g['score']>.5,KalmanPosition(center(g['box'])),g['box'].copy())
            self.next_id+=1;kept.append(t);claimed.add(gi);output.append(self.annotation(t,index,'observed_seed'))
            events.append({'event':'seeded','track_id':t.id})
        # Suppress only observations already represented by an identity. Low-score raw candidates stay raw.
        claimed_boxes=[groups[gi]['box'] for gi in claimed]
        for a in raw:
            if a['object_id'] in EXCLUDED:continue
            if any(iou(np.asarray(a['bbox'])*SIZE,b)>=.65 for b in claimed_boxes):continue
            output.append({**deepcopy(a),'kind':'fresh'})
        self.tracks=kept
        command,target,decision=self.camera(index,level,groups,matches,kept,motion,events)
        self.last=index
        output=sorted(output,key=lambda a:-a['confidence'])[:500]
        diag={'motion':motion,'events':events,'association':association,'tracks':len(kept),'persistent_tracks':sum(t.persistent for t in kept),
         'decision':decision,'target':target,'observed_raw_count':len(raw),'observed_position_updates':len(association),
         'no_hidden_full_view_inputs':True}
        return output,command,diag

    def annotation(self,t,index,kind):
        shown=t.observed_box if t.seen==index and t.observed_box is not None else t.box
        return {'object_id':t.label,'bbox':(clip(shown)/SIZE).tolist(),'confidence':float(t.score),'kind':kind,
         'track_id':t.id,'memory_age':index-t.confirmed,'last_confirmed':t.confirmed,'last_seen':t.seen,
         'persistent':bool(t.persistent),'unseen_age':index-t.seen,'position_uncertainty_l0_px':float(t.uncertainty),
         'kalman_center_l0_px':t.kalman.x[:2].tolist(),'kalman_residual_velocity_l0_px':t.kalman.x[2:].tolist(),'kalman_updates':t.kalman.updates}

    def camera(self,index,level,groups,matches,tracks,motion,events):
        # Propagate cooldown locations with current observed motion, not fixed coordinates.
        if motion['accepted']:
            for old in self.cooldowns:old['box']=warp(old['box'],motion['matrix'])
        self.cooldowns=[c for c in self.cooldowns if index-c['index']<=4]
        if level==1:
            self.l0_streak=0;return {'resolution_level':0,'center_x':1920,'center_y':1080},None,'return_L0'
        self.l0_streak+=1
        if not motion['accepted'] or self.l0_streak<2:return None,None,'hold_L0_motion_or_refresh'
        if any(t.seen!=index for t in tracks):return None,None,'hold_L0_unresolved_identity'
        choices=[]
        # Next-frame motion prediction uses only the just-measured increment, scaled by real index dt.
        dt=motion['dt'];mat=np.asarray(motion['matrix']);nextmat=np.c_[np.eye(2)+(mat[:,:2]-np.eye(2))/dt,mat[:,2]/dt]
        forecasts={}
        for t in tracks:
            predicted_filter=deepcopy(t.kalman);predicted_filter.predict(1,nextmat,True)
            forecasts[t.id]=predicted_filter
        def future(t):
            wh=sides(warp(t.box,nextmat));c=forecasts[t.id].x[:2]
            return np.r_[c-wh/2,c+wh/2]
        for g in groups:
            if not .1<=g['score']<=.5:continue
            b=g['box'];w,h=sides(b)
            if min(w,h)<2 or max(w,h)>100:continue
            if any(t.persistent and iou(b,t.box)>=.3 for t in tracks):continue
            pred=warp(b,nextmat)
            target_track=next((t for t in tracks if t.seen==index and iou(b,t.observed_box)>=.5),None)
            if target_track is not None:pred=future(target_track)
            if not inside(pred,FRAME,3):continue
            if any(np.linalg.norm(center(pred)-center(c['box']))<35 for c in self.cooldowns):continue
            c=center(pred);cx=4*round(float(np.clip(c[0],240,720)));cy=4*round(float(np.clip(c[1],135,405)))
            crop=np.array([cx/4-240,cy/4-135,cx/4+240,cy/4+135])
            if any(not t.persistent and index+1-t.confirmed>=3 and not inside(future(t),crop) for t in tracks):continue
            # Only inspect when off-crop identities have current location evidence and a safe one-frame error bound.
            if any(not inside(future(t),crop) and 2*forecasts[t.id].position_sigma()+max(.5,motion['residual_px'])>max(3.,.28*min(sides(future(t)))) for t in tracks):continue
            tiny=min(2.,max(.5,math.sqrt(256/(w*h))))
            choices.append((g['score']*(1-g['score'])*tiny,g,pred,cx,cy))
        if not choices:return None,None,'hold_L0_no_safe_candidate'
        score,g,pred,cx,cy=max(choices,key=lambda x:x[0]);self.cooldowns.append({'index':index,'box':pred.copy()});self.last_zoom=index
        return {'resolution_level':1,'center_x':int(cx),'center_y':int(cy)}, {'object_id':g['label'],'confidence':g['score'],'priority':float(score),'predicted_bbox':(pred/SIZE).tolist()},'inspect_L1'
