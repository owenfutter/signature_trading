import numpy as np
from data.path import Path
from mean_variance.utils import shuffle, sigwords, word_to_loc, sig_dim
import itertools
import cvxpy as cp
from esig import tosig

class NumpyMethod:
    @classmethod
    def __init__(self, 
                 device='cpu', 
                 order=2):
        """
        Mean-Variance Sig-Trading
        --- Numpy / esig Method ---

        To be passed into the SigTrader class when method = "numpy"
        """
        self.device = device
        self.order = order
        self.l = None
        
    def fit(self, 
        path_train:Path=None,
        asset_paths=None, 
        signal_paths=None,
        fast_sol=True, 
        order=None, 
        verbose=False, 
        max_var=None, 
        precomputed_shuffles = {}):
        """
        Fit the model using the training paths and compute the linear functional l.

        Args:
            path_train (Path object): Train path object and associated methods
            asset_paths (torch.Tensor): Asset training paths (if not passed a Path object).
            signal_paths (torch.Tensor): Signal training paths (if not passed a Path object).
            fast_sol (callable): A function to solve the optimization problem.
            order (int): The order of the signature.
            verbose (bool): Whether to display verbose output.
            max_var (float): Maximum variance threshold.
        """    

        if order is None:
            order = self.order

        if fast_sol:
            order_for_calibration = (order + 1)
            self.order_for_calibration = order_for_calibration
        else:
            order_for_calibration = 2 * (order + 1)
            self.order_for_calibration = order_for_calibration

        # Define path class
        if path_train is None:
            # if only input paths are inputted then create basic path object (no transforms)
            self.path_train = Path(asset_paths=asset_paths, signal_paths=signal_paths, device=self.device, method="numpy")

        if self.path_train.method != "numpy":
            raise ValueError("Method must be 'numpy'")
        
        self.dim = self.path_train.d

        self.path_train.pre_fit(order_for_calibration)
        sig_x_hat_ll = self.path_train.sig_x_hat_ll
        exp_sig_x_hat_ll = self.path_train.unsqueeze_(self.path_train.exp_sig_x_hat_ll,axis=0)

        # Fitting the linear functional l
        self.hidden_dims = range(1 + self.dim, 1 + self.dim + self.path_train.hidden_dim) # define indices
        tradable_dims = range(1, 1 + self.dim)

        def shift_operator(m):
            return m + 1 + self.path_train.d + self.path_train.hidden_dim
        
        """ Creating words and locations of the words within the vector """
        I = list(map(lambda x: (shift_operator(x),), tradable_dims)) # the f(m) terms
        Id = sigwords(order, self.path_train.hidden_dim + self.path_train.d + 1, scalar_term=True) # all w (words)
        words = list(map(lambda x: x[0] + x[1], itertools.product(Id, I))) # concat words and f(m)
        locs = word_to_loc(words, 2*(self.dim + self.path_train.hidden_dim + 1), scalar_term=True) # find those wf(m)'s in the LL signature
        self.locs = locs

        """ Calculating mu_sig """
        mu_sig = exp_sig_x_hat_ll[:,locs] # the expected signature PnL vector
        self.mu_sig = mu_sig[0]
            
        """ Calculating var_sig """
        if fast_sol:
            var_sig = np.cov(sig_x_hat_ll[:, locs].T) # faster (but less accurate) computation
        
        
        self.var_sig = var_sig
        # self.var_sig = (self.var_sig + self.var_sig.T) / 2 # ensuring symmetricity
        
        """
        Computing the linear functional l
        """
        self.dim_l = sig_dim(self.dim + self.path_train.hidden_dim + 1, order)
        
        l = cp.Variable(self.dim_l)
        objective = cp.Maximize(self.mu_sig @ l - cp.quad_form(l, self.var_sig))
        problem = cp.Problem(objective)

        problem.solve()

        # normalize for variance
        normalization = np.sqrt(l.value @ self.var_sig @ l.value / max_var) # normalisation factor to ensure the maximum variance is max_var
        self.l_val = l.value / normalization
        self.normalization = normalization

        self.l = l
        
        self.lin_func_computed = True
        
    def compute_pnl(self, 
                path_test:Path=None, 
                asset_paths=None, 
                signal_paths=None,
                lin_func=None,
                tx_cost=0.0, 
                order=None,
                verbose=False):
        """
        Compute the PnL for the given paths.

        Args:
            paths (torch.Tensor): The input paths.
            tx_cost (float): Transaction cost.
            verbose (bool): Whether to display verbose output.

        Returns:
            float: The computed PnL.
        """
        if order is None:
            order = self.order
        
        if lin_func is None:
            lin_func = self.l_val
        # Define path class
        if path_test is None:
            # if only input paths are inputted then create basic path object (no transforms)
            self.path_test = Path(asset_paths, signal_paths, method='numpy', device=self.device)

        if self.path_test.method != "numpy":
            raise ValueError("Method must be 'numpy'")

       # Ensure we have the add time, lead lag and exp_sig transforms complete
        self.path_test.pre_pnl(order)
        
        l = lin_func.reshape(-1, self.dim).T[:, np.newaxis, np.newaxis]
        sig_x_hat = np.tile(self.path_test.sig_x_hat, (self.dim, 1, 1, 1))

        # Obtain position as linear functional on signature
        position = np.sum(l * sig_x_hat, axis=-1).transpose(1, 2, 0)
        position = position[:, :-1]  # remove the last input as it is not traded on
        
        # Get returns of each asset, missing out the first 2 values to avoid lookahead bias
        rets = np.diff(asset_paths, axis=1)[:, 2:]
        
        # PnL of all paths
        pnl = np.cumsum(position * rets, axis=-2).sum(axis=-1)
        pnl = np.concatenate([np.zeros((pnl.shape[0], 1)), pnl], axis=1)  # add back the first 0 at t=0

        # Net out transaction costs
        tx_costs = np.concatenate([
            np.zeros((position.shape[0], 1)),
            np.sum(np.abs(position[:, 0]) * tx_cost, axis=-1, keepdims=True),
            np.cumsum(np.sum(np.abs(np.diff(position, axis=1)), axis=-1) * tx_cost, axis=-1)
        ], axis=-1)
        
        pnl = pnl - tx_costs

        return position, pnl
    
    