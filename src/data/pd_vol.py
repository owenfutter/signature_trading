import numpy as np
from dataclasses import dataclass
from .base import Midprice

class PDVol(Midprice):
    """
    Underlying asset with path dependent volatility and mean-reverting drift.
    """
    
    def __init__(self, 
                 sigma,
                 theta,
                 OU_theta,
                 alpha=0.2,
                 beta=1,
                 skew_scale=15.8,
                 T=1.,
                 n=100):
        self.sigma = sigma
        self.theta = theta
        self.OU_theta = OU_theta
        self.T = T
        self.n = n
        self.alpha = alpha
        self.beta = beta # beta = 0 is static vol, beta = 1 is path dependent vol
        self.skew_scale = skew_scale
    
    @staticmethod
    def skew(x):
        # return  np.sqrt( (1 - 4*x + 20*(x**2) -15*(x**3) + 20*(x**4)) /100) 
        return np.sqrt(((0.5 + (x-0.2)  + 5*(x-0.2)**2  -10*(x-0.2)**3 + 10*(x-0.2)**4)/100))
    
    @staticmethod
    def ewm(path, alpha=0.2):
        # ema_paths = torch.zeros_like(paths)
        ema_path = np.zeros_like(path)
        ema = 0.  # Initialize EMA
        for j in range(1, path.size):  # Iterate over time steps
            ema = alpha * path[j] + (1 - alpha) * ema
            ema_path[j] = ema
        return ema_path
    
    def generate(self):

        dt = self.T / self.n
        dW = np.random.randn(self.n)

        # Define drift and vol
        MA = self.ewm(dW, alpha=self.alpha)
        # mean reverting drift based on ewm of idiosyncratic noise
        drift = -self.theta*MA
        # define vol as skew transformed drift
        pd_vol = self.skew(self.skew_scale*(MA))
        # add static vol (not path-depdendent) to be scaled via beta
        static_vol = np.mean(pd_vol,axis=0,keepdims=True)
        # time dimension
        time = np.linspace(0, self.T, self.n)

        # Construct terms only dependent on idiosyncratic noise and time
        path = np.zeros(self.n)
        
        for i in range(1, self.n):
            path[i] = path[i-1] + (drift[i-1] - self.OU_theta * path[i-1]) * dt + (self.sigma * ( self.beta*pd_vol[i-1] + (1-self.beta)*static_vol) * dW[i])

        return np.c_[time, 1.+path]
