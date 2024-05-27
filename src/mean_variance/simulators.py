import torch
import numpy as np

def generate_gbm(n_samples:int, n_days:int, mu, vol, dt:torch.float64, dim: int = 1):
    voldt = np.sqrt(dt) * vol
    mudt = dt * mu
    z = torch.randn(n_samples, n_days, dim)
    paths = (voldt * z - voldt**2 / 2. + mudt).cumsum(1).exp()
    x = torch.cat([torch.ones(n_samples, 1, dim), paths], dim=1)
    return x

def generate_multi_gbm(n_samples: int, n_days: int, mu_vec: torch.tensor, cov_mat: torch.tensor, dt: torch.float64, dim: int):
    n_steps = n_days*dt
    
    # case: given correlation matrix, create paths for multiple correlated processes
    if cov_mat.shape[0]>1:
        # paths = torch.zeros(size = (n_samples, n_steps, dim))
        
        # loop through number of paths
        dim = mu_vec.shape[0]

        choleskyMatrix = torch.linalg.cholesky(cov_mat)
        e = torch.normal(0, 1, size=(n_samples, n_steps, dim))
        noise = torch.matmul(e,choleskyMatrix)
        
        paths = torch.cumsum(noise,dim=1)

        for i in range(dim):
            paths[:,:,i] = torch.exp(torch.arange(n_steps)*mu_vec[i]+ paths[:,:,i])
            
        paths = torch.cat([torch.ones(n_samples, 1, dim), paths], dim=1)

    # case: no given correlation matrix, create paths for a single process
    else:
        paths = generate_gbm(n_samples, n_days, mu_vec[0], cov_mat[0], dt, dim)
    return paths

def generate_heston_paths(
    n_samples:int, 
    n_days:int, 
    dt: torch.float64, 
    dim: int, 
    S_0: int, 
    V_0: torch.float64, 
    kappa: torch.float64, 
    theta: torch.float64, 
    nu: torch.float64, 
    rho: torch.float64):

    n_steps = n_days*dt

    # first, generate 2 correlated brownian motion paths

    dW1 = torch.normal(0, 1, size=(n_samples, n_steps, dim))
    dW2 = rho*dW1 + torch.sqrt(1-rho**2)*torch.normal(0, 1, size=(n_samples, n_steps, dim))

    paths = torch.ones(size = (n_samples, n_steps, 2))

    paths[:,0,0] = V_0
    paths[:,0,1] = S_0

    for i in range(1,n_steps):

            paths[:,i,0] = paths[:,i-1,0] + kappa*(theta - paths[:,i-1,0])*dt + nu*torch.sqrt(paths[:,i-1,0])*dW1[:,i-1,0] # Euler discretisation
            paths[:,i,0] = torch.abs(paths[:,i,0]) # Ensuring positive volatility by reflecting
            paths[:,i,1] = paths[:,i-1,1]*torch.exp(-0.5*paths[:,i-1,0]*dt + torch.sqrt(paths[:,i-1,0])*dW2[:,i-1,0]) # Euler discretisation

    return paths

def generate_OU_paths(
    n_samples:int, 
    n_days:int, 
    dt: torch.float64, 
    dim: int, 
    S_0: int,
    theta: torch.float64,
    sigma: torch.float64, 
    mu: torch.float64):

    n_steps = n_days*dt

    dW1 = torch.normal(0, 1, size=(n_samples, n_steps, dim))
    paths = torch.ones(size = (n_samples, n_steps, 1))

    paths[:,0,0] = S_0

    for i in range(1,n_steps):

            paths[:,i,0] = paths[:,i-1,0]+ (theta*(mu - paths[:,i-1,0])*dt + sigma*dW1[:,i-1,0])

    return paths

def generate_2d_OU_paths(
    n_samples: int, 
    n_days: int, 
    dt: torch.float64, 
    dim: int, 
    S_0: int,
    theta: torch.float64,
    sigma: torch.float64, 
    mu: torch.float64,
    rho: torch.float64):  # Correlation coefficient between two dimensions

    n_steps = int(n_days / dt)

    dW = torch.normal(0, 1, size=(n_samples, n_steps, dim))  # Independent Brownian motions

    # Generate correlated Brownian motions
    dW_corr = torch.stack([dW[:, :, 0], rho * dW[:, :, 0] + torch.sqrt(1 - rho**2) * dW[:, :, 1]], dim=-1)

    paths = torch.ones(size=(n_samples, n_steps, dim))
    paths[:, 0, :] = S_0

    for i in range(1, n_steps):
        for j in range(dim):
            paths[:, i, j] = paths[:, i-1, j] + (theta[j] * (mu[j] - paths[:, i-1, j]) * dt + sigma[j] * dW_corr[:, i-1, j])

    return paths


class SignalProcess:

    def __init__(self,
                 lookback,
                 mean_rev,
                 sigma_f,
                 sigma_Z,
                 sigma_X,
                 kernel_lambda,
                 rho,
        dim=1):

        self.mu = 0
        self.lookback = lookback
        self.dim = dim
        self.theta = mean_rev
        self.sigma_f = sigma_f
        self.sigma_Z = sigma_Z
        self.kernel_lambda = kernel_lambda
        self.sigma_X = sigma_X
        self.rho=rho

        self.cov_mat = torch.Tensor([[1,self.rho],[self.rho,1]])
        dim = 2

        self.choleskyMatrix = torch.linalg.cholesky(self.cov_mat)

    def simulate_factor(self, 
                        n_samples,  
                        n_steps):

        # paths = generate_OU_paths(n_samples=n_samples,
        #                                 n_days=n_steps+self.lookback+1,
        #                                 dt=1, 
        #                                 dim=1, 
        #                                 S_0=0,
        #                                 theta = self.theta,
        #                                 sigma = self.sigma_f, 
        #                                 mu=self.mu)

        dW1 = self.noise[:,:,0]
        paths = torch.ones(size = (n_samples, n_steps+self.lookback+1))

        paths[:,0] = 0

        for i in range(1,n_steps):

                paths[:,i] = paths[:,i-1]+ (-self.theta*(paths[:,i-1]) + self.sigma_f*dW1[:,i-1])

        self.factor_paths = paths

    def simulate_latent(self,
                        n_samples,
                        n_steps):
        
        # latent_noise = self.sigma_Z*self.noise[:,:,1].cumsum(1)
        latent_noise = self.sigma_Z*self.noise[:,:,1]
        self.latent_noise = latent_noise

        # normalization = integrate.quad(lambda x: np.exp(-self.kernel_lambda*(x-self.lookback)), 0, self.lookback)[0]

        lb_range = np.arange(self.lookback)
        weights = np.exp(-self.kernel_lambda*(lb_range-self.lookback))
        self.weights = weights/weights.sum()

        self.kernel_factor = (torch.diff(self.factor_paths,dim=1).unfold(dimension=1,size=self.lookback,step=1)[:,:-1,:]*torch.Tensor(self.weights).unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        
        self.latent_paths = self.kernel_factor + self.latent_noise

    def simulate_process(self,
                       n_samples,
                       n_steps,
                       pd_vol=False,
                       beta1 = 1,
                       beta2 = 1):

        e = torch.normal(0, 1, size=(n_samples, n_steps,2))
        self.noise = torch.matmul(e,self.choleskyMatrix)

        self.simulate_factor(n_samples, n_steps)
        self.simulate_latent(n_samples, n_steps)

        dW1 = torch.normal(0, 1, size=(n_samples, n_steps+1))
        paths = torch.ones(size = (n_samples, n_steps+1))

        paths[:,0] = 1

        vols = torch.normal(self.sigma_X, self.sigma_X/2, size = (n_samples, n_steps+self.lookback+1))

        for i in range(1,n_steps):

            if pd_vol == True:
        

                vol_kernel = ((vols**2)[:,i-1:i+self.lookback-1]*torch.Tensor(self.weights).unsqueeze(0)).sum(dim=-1)

                # vols[:,i] = (self.sigma_X*torch.ones(n_samples) - beta1*self.latent_paths[:,i]**2 + beta2*torch.sqrt(vol_kernel))/np.sqrt(252)

                vols[:,i] = (- beta1*self.latent_paths[:,i]**2 + beta2*torch.sqrt(vol_kernel))/np.sqrt(252)

                paths[:,i] = paths[:,i-1]+ self.latent_paths[:,i] + vols[:,i]*dW1[:,i-1]

                self.vol_kernel = vol_kernel

            else: 
                paths[:,i] = paths[:,i-1]+ self.latent_paths[:,i] + self.sigma_X*dW1[:,i-1]

            
        
        self.vols = vols
        
        self.asset_paths = paths

    def simulate_process_hvol(self,
                       n_samples,
                       n_steps,
                        V_0=0.04, 
                        kappa=1, 
                        theta=0.09, 
                        nu=0.4, 
                        rho=-0.8,
                        mu_0 = 0.01):

        e = torch.normal(0, 1, size=(n_samples, n_steps,2))
        self.noise = torch.matmul(e,self.choleskyMatrix)

        self.simulate_factor(n_samples, n_steps)
        self.simulate_latent(n_samples, n_steps)

        paths = torch.ones(size = (n_samples, n_steps+1))

        paths[:,0] = 1

        dWX = torch.normal(0, 1, size=(n_samples, n_steps+1))
        dWV = rho*dWX + np.sqrt(1-rho**2)*torch.normal(0, 1, size=(n_samples, n_steps+1))

        # paths = torch.ones(size = (n_samples, n_steps, 2))
        vols = torch.ones(size = (n_samples, n_steps+1))

        vols[:,0] = V_0
    
        for i in range(1,n_steps):
                        

            vols[:,i] = vols[:,i-1] - self.latent_paths[:,i] + kappa*(theta - vols[:,i-1]) + nu*torch.sqrt(vols[:,i-1])*dWV[:,i-1] # Euler discretisation
            vols[:,i] = torch.abs(vols[:,i]) # Ensuring positive volatility by reflecting

            # paths[:,i,1] = paths[:,i-1,1]*torch.exp(-0.5*paths[:,i-1,0]*dt + torch.sqrt(paths[:,i-1,0])*dW2[:,i-1,0]) # Euler discretisation

            # vols[:,i] = (- beta1*self.latent_paths[:,i]**2 + beta2*torch.sqrt(vol_kernel))/np.sqrt(252)

            paths[:,i] = paths[:,i-1] + mu_0 - torch.sqrt(vols[:,i])/100 + self.latent_paths[:,i] + torch.sqrt(vols[:,i]/252)*dWX[:,i-1]

        # self.vol_kernel = vol_kernel
        
        self.vols = vols
        
        self.asset_paths = paths

class PathDependentSimulator:
    def __init__(self, 
            sigma,
            theta,
            OU_theta,
            rho):
        
        self.sigma = sigma
        self.theta = theta
        self.OU_theta = OU_theta
        self.rho = rho
    
    def skew(self,x):
        # return  np.sqrt( (1 - 4*x + 20*(x**2) -15*(x**3) + 20*(x**4)) /100) 
        return np.sqrt(((0.5 + (x-0.2)  + 5*(x-0.2)**2  -10*(x-0.2)**3 + 10*(x-0.2)**4)/100))

    
    def exponential_moving_average(self,paths, alpha=0.2):
        ema_paths = torch.zeros_like(paths)
        for i in range(paths.size(0)):  # Iterate over paths
            ema = paths[i, 0] - paths[i, 0]  # Initialize EMA with the first observation
            for j in range(1, paths.size(1)):  # Iterate over time steps
                ema = alpha * paths[i, j] + (1 - alpha) * ema
                ema_paths[i, j] = ema
        return ema_paths

    def simulate_paths(self, n_days, dt, n_samples, alpha=0.2, beta=0.8, skew_scale=15.8):

        dim = 2
        n_steps = int(n_days / dt)
        
        dW = torch.normal(0, 1, size=(n_samples, n_steps, dim))  # Independent Brownian motions
        drift = torch.zeros_like(dW)
        # vol = torch.zeros_like(dW)
        
        # Generate correlated Brownian motions
        dW_corr = torch.stack([dW[:, :, 0], self.rho * dW[:, :, 0] + torch.sqrt(1 - self.rho**2) * dW[:, :, 1]], dim=-1)

        MA = self.exponential_moving_average(dW_corr, alpha=alpha)
        drift = -self.theta*MA
        pd_vol = self.skew(skew_scale*(MA))
        static_vol = pd_vol.mean(dim=0,keepdim=True).mean(dim=1)

        paths = torch.zeros(size=(n_samples, n_steps, dim))
        # drift[:, 0, :] = self.theta*torch.normal(0,1,size=(n_samples,dim))
        # vol[:, 0, :] = 1
        # paths[:, 0, :] = self.sigma*vol[:, 0, :] * dW_corr[:, 0, :]

        # dW_uncorr = torch.normal(0, 1, size=(n_samples, n_steps, dim))

        for i in range(1,n_steps):
                
                # drift[:, i, :] = self.exponential_moving_average(paths[:,:i], alpha)[:,-1,:]
                # vol[:,i,:] = self.skew(15.8*drift[:, i, :])

                # paths[:, i, :] = paths[:, i-1, :] + ( drift[:, i-1, :] * dt + self.sigma*vol[:, i-1, :] * (beta*dW_corr[:, i, :] + (1-beta)*dW_uncorr[:, i, :]))
                paths[:, i, :] = paths[:, i-1, :] + ( (drift[:, i-1, :] - self.OU_theta * paths[:, i-1, :] ) * dt) + (self.sigma * ( beta*pd_vol[:, i-1, :] + (1-beta)*static_vol ) * dW_corr[:, i, :])

        self.paths = paths
        self.drift = drift - self.OU_theta * paths
        self.dW_corr = dW_corr
        self.ewm = MA
        self.vol = self.sigma * ( beta*pd_vol + (1-beta)*static_vol.unsqueeze(0) )
        self.noise = self.vol[:,:-1]*self.dW_corr[:,1:]