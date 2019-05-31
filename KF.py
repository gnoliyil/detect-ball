import numpy as np
import time, threading

class KF:
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
        self.mu = mu0
        self.sigma = sigma0
        self.Q = Q
        self.R = R
        self.g = g
        self.delta_t = delta_t
        self.C = C
        self.lock = threading.lock()
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
    
    def keepPredicting(self):
       while True:                                                # this should be set as the end condition 
            time.sleep(self.delta_t)
            self.lock.acquire()
            try:
                self.t += self.delta_t
                self.mu = self.A.dot(self.mu) + self.u
                self.sigma = self.A.dot(self.sigma).dot(self.A.T) + self.Q
                print('One prediction step completed at time {}s.'.format(self.t))
            finally:
                self.lock.release()
                
    def update(self, y):
        """
        Update the KF using a measurement.

        Args:
             y: Measurement.
        """
        self.lock.acquire()
        try:
            Kt = self.sigma * self.C.T * np.linalg.inv(self.C.dot(self.sigma).dot(self.C.T) + self.R)
            self.mu = self.mu + Kt.dot(y - self.C.dot(self.mu))
            self.sigma = self.sigma - Kt.dot(self.C).dot(self.sigma)
            print('One update step completed at time {}s.'.format(self.t))
        finally:
            self.lock.release()
                
    def predictFinalPosition(self,):
        """
        Predict the state of the ball when it reaches the plane of the goal line.

        Returns:
            mu_predicted: Predicted state of the ball.
            sigma_predicted: Prediction error covariance.
        """
        mu_predicted = self.mu
        sigma_predicted = self.sigma
        while True:                                                 # this should be set as the end condition
            mu_predicted = self.A.dot(mu_predicted) + self.u
            sigma_predicted = self.A.dot(sigma_predicted).dot(self.A.T) + self.Q
        return mu_predicted, sigma_predicted
            
            
            