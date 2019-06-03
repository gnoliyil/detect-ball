import numpy as np
import time, threading
import redis

class KF:
    
    BALL_POSITION_KEY  = 'sai2::cs225a::cv::ball_position::xyz'
    BALL_TIMESTAMP_KEY = 'sai2::cs225a::cv::ball_position::time'
     
    def __init__(self, mu0, sigma0, C, Q, R, g, delta_t):             # Todo: add boundary for the end condition of the KF
        """
        Initialize Kalman Filter.

        Args:
             mu0: Initial state mean.
             sigma0: Initial state covariance.
             C: C in the measurement equation.
             Q: Process noise covariance.
             R: Measurement noise covariance.
             g: Gravitational constant.
             delta_t: Time step.
        """
        self.redis = redis.Redis(charset='utf-8', decode_responses=True)
        self.prev_timestamp = None
        
        self.mu = mu0
        self.sigma = sigma0
        self.Q = Q
        self.R = R
        self.g = g
        self.delta_t = delta_t
        self.C = C
        self.lock = threading.Lock()
        self.t = 0
        
        self.A = np.array([[1., 0., 0., delta_t, 0.,      0.     ],
                           [0., 1., 0., 0.,      delta_t, 0.     ],
                           [0., 0., 1., 0.,      0.,      delta_t],
                           [0., 0., 0., 1.,      0.,      0.     ],
                           [0., 0., 0., 0.,      1.,      0.     ],
                           [0., 0., 0., 0.,      0.,      1.     ]])
        self.u = np.array([0, 0, 0, 0, 0, g * delta_t]).reshape((6, 1))
        
    def startKF(self):
        """
        Start a thread that keeps predicting at the interval of the time step.
        
        """
        t = threading.Thread(target=self.keepPredicting, name='PredictThread')
        t.start()
        print('Start predicting at time {}s.'.format(self.t))
    
    def predict(self):
        self.t += self.delta_t
        self.mu = self.A.dot(self.mu) + self.u
        self.sigma = self.A.dot(self.sigma).dot(self.A.T) + self.Q
#         print('One prediction step completed at time {}s.'.format(self.t))
                
    def update(self, y):
        """
        Update the KF using a measurement.

        Args:
             y: Measurement.
        """
#         print(self.sigma.shape)
#         print(self.C.shape)
#         print(self.R.shape)
#         temp = self.C.dot(self.sigma).dot(self.C.T) + self.R
#         print(temp.shape)

        Kt = self.sigma.dot(self.C.T).dot(np.linalg.inv(self.C.dot(self.sigma).dot(self.C.T) + self.R))
        # print(self.mu.shape, self.sigma.shape, Kt.shape)
        # print(y.shape, self.C.dot(self.mu).shape)
        self.mu = self.mu + Kt.dot(y.reshape(3,1) - self.C.dot(self.mu))
        self.sigma = self.sigma - Kt.dot(self.C).dot(self.sigma)
        # print(self.mu.shape, self.sigma.shape, Kt.shape)

#         print('One update step completed at time {}s.'.format(self.t))
                
    def predictFinalPosition(self,):
        """
        Predict the state of the ball when it reaches the plane of the goal line.

        Returns:
            mu_predicted: Predicted state of the ball.
            sigma_predicted: Prediction error covariance.
        """
        ground = -0.5
        # print(self.mu[5] ** 2 - 2 * self.g * (self.mu[2] - ground))
        # print(self.mu[2])
        t = (- self.mu[5, 0] - np.sqrt(self.mu[5, 0] ** 2 - 2 * self.g * (self.mu[2, 0] - ground))) / self.g
        final_position = [self.mu[0, 0] + self.mu[3, 0] * t, self.mu[1, 0] + self.mu[4, 0] * t, 0]
        return final_position
            
    
    def _getTimeStampFromRedis(self):
        return float(self.redis.get(self.BALL_TIMESTAMP_KEY))
        
    def _getPositionFromRedis(self):
        s = self.redis.get(self.BALL_POSITION_KEY)
        return np.array([float(x) for x in s[1:-1].split(',')])
        
    def newPositionOrNone(self):
        new_ts = self._getTimeStampFromRedis()
        pos = self._getPositionFromRedis()
        if new_ts != self.prev_timestamp and pos[2] > 0:
            
            self.prev_timestamp = new_ts
            return pos, new_ts
        else:
            return None, None
            
    def mainLoop(self):
    	numfound = 0
        while numfound < 2:
            new_position_maybe, ts = self.newPositionOrNone()
            if new_position_maybe is not None:
                numfound += 1
                new_pos = np.array([new_position_maybe[0], new_position_maybe[2],-new_position_maybe[1]]) 
                if numfound == 1:
            	    pos1 = new_pos
            	    ts0 = ts

                if numfound == 2:
            	    vel2 = (new_pos - pos1) / (ts - ts0)
            	    self.mu[:3] = new_pos.reshape(3,1)
            	    self.mu[3:] = vel2.reshape(3,1)
                    self.t = ts

        # print("here")    	
        while True:
            new_position_maybe, ts = self.newPositionOrNone()
            if new_position_maybe is not None:
            	# started = True
            	while self.t < ts:
                    self.predict()  
                new_pos = np.array([new_position_maybe[0], new_position_maybe[2],-new_position_maybe[1]])                	
                print("camera", new_pos)
                self.update(new_pos)
                m_p = self.predictFinalPosition()
                # print(new_pos)
                print("predict", m_p[:3])
            
            
                
                
if __name__ == "__main__":
    m0 = np.array([-1.13653743,  2.20493359,  0.0318083, 1, 0, 0.5])

    kf = KF(mu0=m0.reshape(6,1), sigma0=0.1*np.eye(6), 
            C=np.hstack((np.eye(3), np.zeros((3, 3)))), Q=0.1*np.eye(6),
	    R=0.1*np.eye(3), g=-9.8, delta_t=0.01)
    kf.mainLoop()

    #x, z, -y