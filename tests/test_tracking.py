import numpy as np
import lifecycle_helpers as lifecycle
from tracking import MemoryCameraPolicy
from kalman_position import KalmanPosition

def test_regression():
    lifecycle.MemoryCameraPolicy=MemoryCameraPolicy
    lifecycle.run()
    k=KalmanPosition([100.,100.])
    for i in range(1,12):
        # Camera moves 5px/frame, object has another 2px/frame of residual motion.
        k.predict(1,[[1,0,5],[0,1,0]],True)
        before=k.position_sigma();k.update([100+7*i,100],.8)
        assert k.position_sigma()<before
        assert np.linalg.eigvalsh(k.P).min()>-1e-8
    assert abs(k.x[2]-2)<.15,k.x
    expected=100+7*12
    assert abs(k.predict(1,[[1,0,5],[0,1,0]],True)[0]-expected)<.3
    p=MemoryCameraPolicy(lifecycle.Motion())
    lifecycle.step(p,0,[])
    _,command,_=lifecycle.step(p,1,[lifecycle.det((40,200,55,215),.8),lifecycle.det((600,250,610,260),.2,'hangar')])
    assert command is None # predicted velocity covariance makes off-crop carry unsafe
    print('PASS: Kalman measurement covariance contraction, residual velocity convergence, stable PSD covariance, one-step forecast')
if __name__=='__main__':test_regression()

import unittest

class PolicyRegression(unittest.TestCase):
    def test_policy(self):
        test_regression()
