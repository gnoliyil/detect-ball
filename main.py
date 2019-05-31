import numpy as np
import time
import random
from find_ball import FindBall
from KF import KF

if __name__ == "__main__":
    fb = FindBall()
#    kf = KF(mu0 = np.zeros((6,1)), sigma0 = 0.1 * np.eye(6), 
#             C = np.hstack((np.eye(3), np.zeros((3, 3)))), Q = 0.1 * np.eye(6), 
#             R = 0.1 * np.eye(6), g = -9.8, delta_t = 0.1)
#    kf.startKF()
    tot_frames = 0
    time_old = time.time()
    while True:
        try:
            r, x, y, z = fb.find_ball()
            # r, x, y, z = np.random.rand(4)            
            # if r < 0.10: 
            #     print("update! [{}, {}, {}], r = ".format(x, z, -y, r))
            #     kf.update(np.array([x, z, -y]))
            # time.sleep(0.025)
            tot_frames += 1
            print('done')    
        except Exception as ex:
            print('no ball')
            pass
        except KeyboardInterrupt:
            break
    time_now = time.time()
    print(tot_frames / (time_now - time_old))
    
