import numpy as np
import torch
import signatory
import statsmodels.api as sm

from mean_variance.utils import *
from mean_variance.simulators import *
from mean_variance.sig_trader import *
from data.path import Path


class SigFactorModel(SigTrader):
    r"""
    Sig Factor Model Class

    This class is a subclass of the SigTrader class. It is used to fit a linear factor model to the signature paths, 
    and perform a one-period markowtiz optimisation
    """
    def __init__(self, ST):
        """
        Initialise with a SigTrader instance, to access previously computed values and methods
        """
        self.ST = ST
        self.order = ST.order
        self.train_paths = ST.strategy.path_train.asset_paths

    def compute_signature_paths(self, asset_paths=None, order=None):
        """
        Signature functionality
        """
        if asset_paths is None:
            asset_paths = self.train_paths
        if order is None:
            order = self.order
        
        x_hat = self.ST.strategy.path_train._add_time_pytorch(asset_paths)
        
        path_class = signatory.Path(x_hat, order, scalar_term=True, basepoint=False)
        sig_x_hat = self.ST.strategy.path_train.backend.cat(
                    [path_class.signature(0, i).unsqueeze(1) for i in range(2, x_hat.shape[1])],
                    dim=1)
                
        return sig_x_hat
    
    def fit(self, max_var = 0.05):
        """
        1. Fit the linear model for each asset
            \hat{y} = \langle ell, Sig(X) \rangle
        
        2. Apply the dxd inverse covariance matrix to the y_hats to get the strategy rets
        
        """

        train_paths = self.train_paths

        B = torch.Tensor([])
        resid = torch.Tensor([])

        X_input = self.compute_signature_paths(self.train_paths)
        X_input = X_input[:,:-1,:]
        y_input = train_paths.diff(dim=1)[:,-X_input.shape[1]:]

        for asset in range(train_paths.shape[-1]):
    
            xy_data = torch.cat([X_input.flatten(end_dim=1), y_input[:,:,asset:asset+1].flatten(end_dim=1)], dim=-1)
            xy_data = xy_data[~torch.any(xy_data.isnan(),dim=-1)]

            x_train = xy_data[:,:-1] # Adds an intercept term to the simple linear regression formula
            y_train = xy_data[:,-1:]
            x = np.array(x_train)
            y = np.array(y_train)

            # Obtain B
            B_model = sm.OLS(y, x)
            B_results = B_model.fit()

            B = torch.cat([B, torch.Tensor(B_results.params).unsqueeze(0)],dim=0).type(torch.float64)
            resid = torch.cat([resid, torch.Tensor(B_results.resid).unsqueeze(0)], dim=0).type(torch.float64)

        Sigma = torch.cov(resid).type(torch.float64)

        lin_func = torch.inverse(Sigma)@B
        self.lin_func = lin_func.T.flatten()
        self.lin_func = self.lin_func.to(self.ST.strategy.var_sig.dtype)

        # Option 2: Convert self.ST.strategy.var_sig to the dtype of self.lin_func
        self.ST.strategy.var_sig = self.ST.strategy.var_sig.to(self.lin_func.dtype)

        # Now perform the operation
        normalization = (self.lin_func @ self.ST.strategy.var_sig @ self.lin_func / max_var).sqrt()

        # Normalize lin_func
        self.lin_func = self.lin_func / normalization
        self.normalization = normalization
        
        self.l = lin_func

        return lin_func

    def compute_pnl(self, asset_paths=None, order=None):

        """
        Compute PnL using the SigTrader instance 
        """

        if order is None:
            order = self.order

        position, pnl = self.ST.compute_pnl(path_test = asset_paths, order=order, lin_func = self.lin_func)

        return position, pnl