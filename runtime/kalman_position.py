"""Per-object center/residual-velocity Kalman filter in the 960x540 source plane."""
import numpy as np

class KalmanPosition:
    def __init__(self,center):
        self.x=np.r_[np.asarray(center,dtype=float),0.,0.]
        self.P=np.diag([1.,1.,4.,4.]);self.updates=0
    def predict(self,dt,matrix,reliable):
        m=np.asarray(matrix,dtype=float) if reliable else np.array([[1.,0.,0.],[0.,1.,0.]])
        A=m[:,:2];F=np.zeros((4,4));F[:2,:2]=A;F[:2,2:]=dt*A;F[2:,2:]=A
        self.x=F@self.x+np.r_[m[:,2],0.,0.]
        G=np.array([[dt*dt/2,0],[0,dt*dt/2],[dt,0],[0,dt]])
        Q=.35**2*(G@G.T)
        if not reliable:Q[:2,:2]+=np.eye(2)*(12*dt)**2
        self.P=F@self.P@F.T+Q
        return self.x[:2].copy()
    def update(self,center,confidence):
        H=np.c_[np.eye(2),np.zeros((2,2))]
        sigma=.5+1.0*(1-float(confidence));R=np.eye(2)*sigma*sigma
        innovation=np.asarray(center)-H@self.x;S=H@self.P@H.T+R
        K=np.linalg.solve(S,(self.P@H.T).T).T
        self.x+=K@innovation
        I=np.eye(4);self.P=(I-K@H)@self.P@(I-K@H).T+K@R@K.T
        self.P=(self.P+self.P.T)/2;self.updates+=1
        return {'innovation_px':innovation.tolist(),'posterior_center_px':self.x[:2].tolist(),
                'residual_velocity_px':self.x[2:].tolist(),'position_sigma_px':self.position_sigma(),
                'updates':self.updates}
    def position_sigma(self):return float(np.sqrt(max(0.,np.linalg.eigvalsh(self.P[:2,:2]).max())))
