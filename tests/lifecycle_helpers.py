"""Targeted position correction, exit, causal motion and camera contracts."""
import numpy as np
import cv2
from tracking import MemoryCameraPolicy,SIZE,FRAME
from visible_view_motion import VisibleViewMotion

class Motion:
    def __init__(self,dx=0.,dy=0.,accepted=True):self.dx=dx;self.dy=dy;self.accepted=accepted;self.last=None
    def estimate(self,index,region,image):
        dt=1 if self.last is None else index-self.last;self.last=index
        return {'accepted':self.accepted,'matrix':[[1.,0.,self.dx*dt],[0.,1.,self.dy*dt]],'residual_px':.2,'dt':dt,'reason':'test_fixture'}

def det(box=(300,200,320,220),score=.8,label='tank'):
    return {'bbox':(np.asarray(box)/SIZE).tolist(),'confidence':score,'object_id':label}
def step(p,index,raw,level=0,region=(0,0,3840,2160)):
    return p.step(index,level,region,raw,np.zeros((540,960,3),np.uint8))
def tracked(out):return [a for a in out if 'track_id' in a]

def run():
    p=MemoryCameraPolicy(Motion())
    a=tracked(step(p,0,[det((300,200,306,206))])[0])[0]
    # No IoU overlap, but small same-class center displacement: correct position and preserve ID.
    b=tracked(step(p,1,[det((306,200,312,206),.4)])[0])[0]
    assert a['track_id']==b['track_id'] and b['bbox']==det((306,200,312,206),.4)['bbox'] and b['last_seen']==1
    # Visible missing object is never blindly emitted; retire after two misses.
    assert not tracked(step(p,2,[])[0]);assert not tracked(step(p,3,[])[0]);assert not p.tracks
    # Fully exiting each edge disappears immediately; un-clipped state is used.
    for box,velocity in [((951,200,959,220),(12,0)),((1,200,9,220),(-12,0)),((300,531,320,539),(0,12)),((300,1,320,9),(0,-12))]:
        p=MemoryCameraPolicy(Motion(*velocity));step(p,0,[det(box)])
        out,_,diag=step(p,1,[])
        assert not tracked(out) and any(e['event']=='left_frame' for e in diag['events'])
    # Partial frame edge observation must remain, with corrected current bounds.
    p=MemoryCameraPolicy(Motion(5,0));step(p,0,[det((940,200,959,220))])
    assert tracked(step(p,1,[det((945,200,960,220),.6)])[0])
    # Outside L1 != outside full frame; preserve reliable one-frame position.
    p=MemoryCameraPolicy(Motion(2,0));step(p,0,[det((40,200,70,230))])
    a=tracked(step(p,1,[],1,(960,540,2880,1620))[0])[0]
    assert a['kind']=='memory_prediction' and a['last_seen']==0 and abs(a['bbox'][0]*960-42)<1e-5
    b=tracked(step(p,2,[det((45,200,75,230),.25)])[0])[0]
    assert b['track_id']==a['track_id'] and b['last_seen']==2 and abs(b['bbox'][0]*960-45)<1e-5
    # Failed motion cannot output stale geometry outside the crop.
    p.motion_engine.accepted=False
    assert not tracked(step(p,3,[],1,(960,540,2880,1620))[0])
    # Even high-confidence identities cannot emit unobserved geometry beyond two frames.
    p=MemoryCameraPolicy(Motion(1,0));step(p,0,[det((40,200,100,260))])
    step(p,1,[],1,(960,540,2880,1620));step(p,2,[],1,(960,540,2880,1620))
    assert not tracked(step(p,3,[],1,(960,540,2880,1620))[0])
    # Strict confidence split, and no upgrades from propagated outputs.
    p=MemoryCameraPolicy(Motion());a=tracked(step(p,0,[det(score=.5)])[0])[0];assert not a['persistent']
    step(p,1,[det(score=.2)]);step(p,2,[det(score=.2)])
    out=step(p,3,[det(score=.2)])[0]
    assert not tracked(out) and any(a['kind']=='fresh' and a['confidence']==.2 for a in out)
    # Ambiguous symmetric neighbor must not steal an old identity.
    p=MemoryCameraPolicy(Motion());old=tracked(step(p,0,[det((300,200,310,210))])[0])[0]['track_id']
    out,_,_=step(p,1,[det((295,200,305,210),.4),det((305,200,315,210),.4)])
    assert not any(a.get('track_id')==old for a in out)
    # dt follows true index gap (the saved sequence misses index001).
    p=MemoryCameraPolicy(Motion(2,0));step(p,0,[det((40,200,100,260))])
    a=tracked(step(p,2,[],1,(960,540,2880,1620))[0])[0]
    assert abs(a['bbox'][0]*960-44)<1e-5 and a['unseen_age']==2
    # Image motion estimates camera-plane displacement across actual L0/L1 inputs.
    rng=np.random.default_rng(42)
    image=rng.integers(0,256,(540,960,3),dtype=np.uint8)
    image=cv2.GaussianBlur(image,(5,5),1)
    moved=cv2.warpAffine(image,np.float32([[1,0,7],[0,1,5]]),(960,540))
    motion=VisibleViewMotion();assert not motion.estimate(0,(0,0,3840,2160),image)['accepted']
    crop=cv2.resize(moved[135:405,240:720],(960,540),interpolation=cv2.INTER_LINEAR)
    m=motion.estimate(1,(960,540,2880,1620),crop)
    assert m['accepted'] and np.allclose(np.asarray(m['matrix'])[:,2],[7,5],atol=.5),m
    print('PASS: correction without IoU, low confidence updates, visible ghosts, all four exits, partial edge, crop-outside memory, motion failure, bounded extrapolation, ambiguity, dt gap, observed-only optical geometry')

if __name__=='__main__':run()
