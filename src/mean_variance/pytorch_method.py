import torch
from data.path import Path 
from mean_variance.utils import shuffle, sigwords, word_to_loc
import itertools

class PyTorchMethod:
    """
    Mean-Variance Sig-Trading
    --- PyTorch / signatory Method ---

    To be passed into the SigTrader class when the method = "pytorch"
    """
    def __init__(self, 
                 device='cpu', 
                 order=2):
        """
        Initialize the PyTorchMethod with universal parameters.

        Args:
            device (str): The device to use ('cpu' or 'cuda').
            dim (int): The dimension of the input data.
            order (int): The order of the signature.
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
            self.order_for_calibration = (order + 1)
        else:
            self.order_for_calibration = 2 * (order + 1)

        # Define path class
        if path_train is None:
            # if only input paths are inputted then create basic path object (no transforms)
            self.path_train = Path(asset_paths=asset_paths, signal_paths=signal_paths, device=self.device, method="pytorch")

        if self.path_train.method != "pytorch":
            raise ValueError("Method must be 'pytorch'")
        
        self.dim = self.path_train.d

        # Ensure we have the add time, lead lag and exp_sig transforms complete
        self.path_train.pre_fit(self.order_for_calibration)
        sig_x_hat_ll = self.path_train.sig_x_hat_ll
        exp_sig_x_hat_ll = self.path_train.exp_sig_x_hat_ll.unsqueeze(0)

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
        self.mu_sig = mu_sig.T.to(self.device)
            
        """ Calculating var_sig """
        if fast_sol:
            var_sig = sig_x_hat_ll[:, locs].T.cov() # faster (but less accurate) computation
        else:
            words_x_words = list(itertools.product(words, words)) # all pairs of words
            locs_x_locs = itertools.product(locs, locs) # all pairs of corresponding locations
            var_sig = torch.zeros(1, exp_sig_x_hat_ll.shape[-1], exp_sig_x_hat_ll.shape[-1]).to(self.device) # initialise the signature PnL covariance matrix

            for word_prod, (loc_1, loc_2) in zip(words_x_words, locs_x_locs):
                locs_shuffle = precomputed_shuffles.get(word_prod) # check if the shuffle has already been computed
                if locs_shuffle is None:
                    locs_shuffle = precomputed_shuffles.get((word_prod[1], word_prod[0])) # check if the shuffle has already been computed
                if locs_shuffle is None:
                    locs_shuffle = word_to_loc(list(shuffle(*word_prod)), dim=2 * (self.dim + self.path_train.hidden_dim + 1), scalar_term=True) # compute the shuffle
                    precomputed_shuffles[word_prod] = locs_shuffle # store the shuffle

                var_sig[:, loc_1, loc_2] = (exp_sig_x_hat_ll[:, locs_shuffle].sum() - exp_sig_x_hat_ll[:, loc_1] * exp_sig_x_hat_ll[:, loc_2]) 
        
            var_sig = var_sig[0][locs][:,locs] # the signature PnL covariance matrix
        
        self.var_sig = var_sig.to(self.device)
        self.var_sig = (self.var_sig + self.var_sig.T) / 2 # ensuring symmetricity
        
        """
        Computing the linear functional l
        """
        l = (torch.inverse(self.var_sig) @ self.mu_sig).T # the optimal linear functional as defined in the paper
        normalization = (l[0] @ self.var_sig @ l[0] / max_var).sqrt() # normalisation factor to ensure the maximum variance is max_var
        self.l = l / normalization
        self.normalization = normalization

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
            lin_func = self.l
        # Define path class
        if path_test is None:
            # if only input paths are inputted then create basic path object (no transforms)
            self.path_test = Path(asset_paths, signal_paths, method='pytorch', device=self.device)

        if self.path_test.method != "pytorch":
            raise ValueError("Method must be 'pytorch'")

        # Ensure we have the add time, lead lag and exp_sig transforms complete
        self.path_test.pre_pnl(self.order)
        
        l = lin_func.reshape(-1, self.dim).T.unsqueeze(1).unsqueeze(1)
        sig_x_hat = self.path_test.sig_x_hat.unsqueeze(0).repeat(self.dim, 1, 1, 1)

        # Obtain position as linear functional on signature
        position = (l * sig_x_hat).sum(-1).permute(1, 2, 0)
        position = position.clone()[:,:-1] # remove the last input as it is not traded on
        
        # Get returns of each asset, missing out the first 2 values to avoid lookahead bias
        rets = torch.diff(asset_paths,dim=1)[:,2:]
        
        # PnL of all paths
        pnl = (position*rets).cumsum(-2).sum(-1)
        pnl = torch.cat([torch.zeros(pnl.shape[0],1).to(self.device),pnl],dim=1) # add back the first 0 at t=0

        # Net out transaction costs
        tx_costs = torch.cat([torch.zeros(position.shape[0],1),
                              (torch.abs(position[:,0])*tx_cost).sum(dim=-1,keepdim=True),
                              torch.abs(torch.diff(position,dim=1)).sum(dim=-1).cumsum(-1)*tx_cost],
                              dim=-1)
        pnl = (pnl.clone() - tx_costs)
    

        return position, pnl