from data.path import Path 
import numpy as np

class SigTrader:
    """
    Mean-Variance Sig-Trading

    Corresponding code infrastructure to the paper 
    "Signature Trading: A Path-Dependent Extension of the Mean-Variance Framework with Exogenous Signals"
    ----------
    
    This is a parent class that takes an input of two different methods (PyTorch or Numpy, whichever you prefer)

    Functionality:
    ----------
        - Given a batch of sample paths, fit the optimal mean-variance Sig-Trading strategy
        - Given an optimal strategy (linear functional) and a test path(s), compute the PnL

    Methods:
    ----------
    compute_expected_signature: 
        Compute the expected signature of the batches of input train paths
    fit:
        Fit the optimal linear functional with respect to the closed form solution
    compute_pnl:
        For given test path(s), compute the PnL of the strategy
        
    Parameters
    ----------
    dim: int
        dimension of the incoming path / time series
    order: int
        order of truncation of the signature
    device: str
        device on which optimisation is performed. default: 'cpu'
    
    """
    def __init__(self,
                 device='cpu',
                 order=2,
                 method='pytorch'):
        
        self.order, self.device = order, device

        self.method = method.lower()
        self.strategy = self._select_strategy()

    def _select_strategy(self):
        if self.method == 'pytorch':
            from .pytorch_method import PyTorchMethod
            return PyTorchMethod(self.device, self.order)
        elif self.method == 'numpy':
            from .numpy_method import NumpyMethod
            return NumpyMethod()
        else:
            raise ValueError("Unsupported method. Choose 'pytorch' or 'numpy'.")

    def fit(self, 
            path_train:Path=None,
            asset_paths=None, 
            signal_paths=None,
            fast_sol=True, 
            order=None, 
            verbose=False, 
            max_var=0.05, 
            precomputed_shuffles = {}):
        
        self.strategy.fit(path_train,
            asset_paths, 
            signal_paths,
            fast_sol, 
            order, 
            verbose, 
            max_var, 
            precomputed_shuffles)
        
        self.mu_sig = self.strategy.mu_sig
        self.var_sig = self.strategy.var_sig
        self.l = self.strategy.l
    
    def compute_pnl(self, 
                    path_test:Path=None, 
                    asset_paths=None, 
                    signal_paths=None,
                    lin_func=None,
                    tx_cost=0.0, 
                    order=None,
                    verbose=False):
        
        return self.strategy.compute_pnl(path_test, 
                    asset_paths, 
                    signal_paths,
                    lin_func,
                    tx_cost,
                    order, 
                    verbose)

    