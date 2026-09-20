"""Estimate source-plane motion from only successive received views, never hidden pixels."""
import cv2
import numpy as np

class VisibleViewMotion:
    def __init__(self):
        self.previous=None

    def estimate(self,index,region,bgr):
        assert bgr.shape==(540,960,3)
        assert all(v%4==0 for v in region)
        x1,y1,x2,y2=[v//4 for v in region]
        gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        canvas=np.zeros((540,960),np.uint8)
        canvas[y1:y2,x1:x2]=cv2.resize(gray,(x2-x1,y2-y1),interpolation=cv2.INTER_AREA)
        valid=np.zeros_like(canvas);valid[y1+6:y2-6,x1+6:x2-6]=255
        prev=self.previous;self.previous=(index,canvas,valid)
        result={'accepted':False,'reason':'no_previous_view','pairs':0,'inliers':0,'dt':None,
                'matrix':[[1.,0.,0.],[0.,1.,0.]],'residual_px':None,'units':'960x540 source-plane pixels'}
        if prev is None:return result
        last,old,oldvalid=prev;dt=index-last;result['dt']=dt
        if not 1<=dt<=2:result['reason']='frame_gap';return result
        mask=cv2.bitwise_and(oldvalid,valid)
        ys,xs=np.nonzero(mask)
        if len(xs)<2000:result['reason']='too_little_overlap';return result
        # Work on the shared rectangle rather than black-padded canvases: otherwise
        # the artificial crop border corrupts coarse optical-flow pyramid levels.
        x0,xend=int(xs.min()),int(xs.max()+1);y0,yend=int(ys.min()),int(ys.max()+1)
        old=old[y0:yend,x0:xend];canvas=canvas[y0:yend,x0:xend]
        valid=valid[y0:yend,x0:xend];mask=mask[y0:yend,x0:xend]
        mask=cv2.erode(mask,np.ones((13,13),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
        points=cv2.goodFeaturesToTrack(old,500,.01,8,mask=mask,blockSize=7)
        if points is None or len(points)<20:result['reason']='few_overlap_features';return result
        nxt,status,_=cv2.calcOpticalFlowPyrLK(old,canvas,points,None,winSize=(31,31),maxLevel=3)
        if nxt is None:result['reason']='flow_failed';return result
        back,back_status,_=cv2.calcOpticalFlowPyrLK(canvas,old,nxt,None,winSize=(31,31),maxLevel=3)
        if back is None:result['reason']='backflow_failed';return result
        p=points[:,0,:];q=nxt[:,0,:];b=back[:,0,:]
        ok=(status[:,0]>0)&(back_status[:,0]>0)&(np.linalg.norm(p-b,axis=1)<1.2)&np.isfinite(q).all(axis=1)
        height,width=canvas.shape
        xi=np.clip(np.rint(q[:,0]),0,width-1).astype(int);yi=np.clip(np.rint(q[:,1]),0,height-1).astype(int)
        ok&=(q[:,0]>=0)&(q[:,0]<width)&(q[:,1]>=0)&(q[:,1]<height)&(valid[yi,xi]>0)
        p=p[ok];q=q[ok];result['pairs']=len(p)
        if len(p)<20:result['reason']='few_consistent_features';return result
        p=p+np.array([x0,y0]);q=q+np.array([x0,y0])
        cv2.setRNGSeed(73)
        mat,inliers=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=2000,confidence=.99,refineIters=10)
        if mat is None:result['reason']='ransac_failed';return result
        inliers=inliers[:,0].astype(bool);n=int(inliers.sum());result['inliers']=n
        err=np.linalg.norm(p@mat[:,:2].T+mat[:,2]-q,axis=1)
        median=float(np.median(err[inliers]));p95=float(np.percentile(err[inliers],95))
        scale=float(np.hypot(mat[0,0],mat[1,0]));angle=float(np.degrees(np.arctan2(mat[1,0],mat[0,0])))
        mid=np.array([480.,270.]);displacement=float(np.linalg.norm(mat[:,:2]@mid+mat[:,2]-mid))
        span=np.ptp(p[inliers],axis=0)
        result.update(residual_px=p95,median_error_px=median,scale=scale,angle_degrees=angle)
        if n<20 or n/len(p)<.55 or median>1.2 or p95>2.5 or min(span)<40:
            result['reason']='weak_consensus';return result
        if not .97**dt<=scale<=1.03**dt or abs(angle)>3*dt or displacement>60*dt:
            result['reason']='implausible_global_motion';return result
        result.update(accepted=True,reason='visible_overlap_flow',matrix=mat.tolist())
        return result
