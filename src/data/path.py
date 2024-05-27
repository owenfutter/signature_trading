import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

class Path:
    """
    A class to handle and process asset and signal paths for signature computations.

    This class provides functionality to concatenate asset and signal paths, add time,
    apply the Hoff lead-lag transformation, compute signatures and expected signatures,
    and normalize paths. It supports both NumPy and PyTorch backends.
    """
    def __init__(self, asset_paths, signal_paths=None, device=None, method=None):
        """
        Initialize the Path object with asset and signal paths.

        Args:
            asset_paths (np.ndarray or torch.Tensor): The asset paths of shape [num_paths, path_length, d].
            signal_paths (np.ndarray or torch.Tensor): The signal paths of shape [num_paths, path_length, N].
            method (str): The method to use ('numpy' or 'pytorch'). Defaults to 'numpy'.
            device (str): The device to use ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")  # Select GPU device
                print("Using GPU for computations.")
            else:
                device = torch.device("cpu")   # Fall back to CPU
                print("CUDA is not available. Using CPU for computations.")
        self.device = device

        if method is None:
            if isinstance(asset_paths, np.ndarray):
                self.method = 'numpy'
            elif torch.is_tensor(asset_paths):
                self.method = 'pytorch'
            else:
                raise ValueError("Unsupported input types. Provide numpy arrays or pytorch tensors.")

        self.method = method

        if self.method == "numpy":
            import iisignature
            self.backend = np
            self.iisignature = iisignature
        if self.method == "pytorch":
            import signatory
            self.backend = torch
            self.signatory = signatory
        
        self.asset_paths = asset_paths
        self.signal_paths = signal_paths
        self.d = self.asset_paths.shape[-1]
        if self.signal_paths is None:
            self.hidden_dim = 0
        else:
            self.hidden_dim = self.signal_paths.shape[-1]
        self.time_added = False

        if signal_paths is None:
            self.paths = asset_paths
        else:
            self.paths = self._concatenate_paths(asset_paths, signal_paths)

    def _concatenate_paths(self, paths_a, paths_b):
        """
        Concatenate two sets of paths along the last dimension.
        """
        return self.backend.concatenate([paths_a, paths_b], axis=-1)
    
    def copy_(self,paths):
        if self.method == "pytorch":
            return paths.clone()
        if self.method == "numpy":
            return paths.copy()
        else:
            raise ValueError("Unsupported method")

    def unsqueeze_(self,paths, axis):
        if self.method == "pytorch":
            return paths.unsqueeze(dim=axis)
        if self.method == "numpy":
            return np.expand_dims(paths, axis=axis)
        else:
            raise ValueError("Unsupported method")


    def add_time(self):
        """
        Add time as the 0-th dimension to the paths.
        -> To be inputted used as add_time_pytorch or add_time_numpy
        """
        if self.time_added:
            print("Time already added")
            return
        elif self.method == 'pytorch':
            self.paths = self._add_time_pytorch(self.paths)
            self.time_added = True
        elif self.method == 'numpy':
            self.paths = self._add_time_numpy(self.paths)
            self.time_added = True

    def _add_time_pytorch(self, x):
        size, length, dim = x.shape
        time_vector = self.backend.linspace(0, 1, length).reshape(1, -1, 1).repeat(size, 1, 1).to(x.device)
        x_hat = self.backend.cat([time_vector, x], dim=2)  # ensures that time is the zero-th index
        return x_hat

    def _add_time_numpy(self, x):
        size, length, dim = x.shape
        time_vector = self.backend.linspace(0, 1, length).reshape(1, -1, 1).repeat(size, axis=0)
        x_hat = self.backend.concatenate([time_vector, x], axis=2)  # ensures that time is the zero-th index
        return x_hat

    @staticmethod
    def _sig(path, order, method, backend, iisignature=None, signatory=None):
        """
        Compute the signature of a batch of paths.

        Args:
            path (np.ndarray or torch.Tensor): The input paths.
            order (int): The order of the signature.
            method (str): The method to use ('numpy' or 'pytorch').
            backend (module): The backend module (numpy or torch).
            iisignature (module, optional): The iisignature module for NumPy.
            signatory (module, optional): The signatory module for PyTorch.

        Returns:
            np.ndarray or torch.Tensor: The computed signature.
        """        
        if method == 'numpy':
            return backend.r_[1., iisignature.sig(path, order)]
        elif method == 'pytorch':
            return signatory.signature(path, order, scalar_term=True)
        else:
            raise ValueError("Unsupported method")

    def compute_T_signature(self, order, n_jobs=-1, verbose=False):
        """
        Compute the signature of all paths stored as self.paths
        -> This is for *fitting* and should be done at terminal time on the LL path
            Output shape: [num_paths, num_sig_terms]
        """
        if self.paths.shape[-1] == self.d + self.hidden_dim or not self.time_added:
            self.add_time()
        
        if self.method == 'pytorch':
            self.signatures = self.signatory.signature(self.paths, order, scalar_term=True)
        elif self.method == 'numpy':
            iterator = tqdm(self.paths, desc="Computing signatures") if verbose else self.paths
            self.signatures = Parallel(n_jobs=n_jobs)(delayed(self._sig)(path, order, self.method, self.backend, self.iisignature) for path in iterator)
            self.signatures = np.array(self.signatures)

    def compute_expected_signature(self, order, n_jobs=-1, verbose=False):
        """
        Compute the expectation of self.signatures
            Output shape: [num_sig_terms]
        """
        self.compute_T_signature(order, n_jobs, verbose)
        self.expected_signature = self.signatures.mean(axis=0)
        return self.expected_signature
    
    def compute_signature_paths(self, order, n_jobs=-1, verbose=False):
        """
        This is to compute the signature for each t from 0 to T along the paths
            Output shape: [num_paths, path_length-2, num_sig_terms]
        """
        if self.paths.shape[-1] == self.d + self.hidden_dim or not self.time_added:
            self.add_time()
        
        # compute the signature of all the paths (batches)
        if self.method == 'pytorch':
            if order == 0:
                    self.sig_x_hat = self.backend.ones(self.paths.shape[0], self.paths.shape[1] - 2, 1) # if order = 0, then the signature is just 1s
            else: 
                path_class = self.signatory.Path(self.paths, order, scalar_term=True, basepoint=False)
                self.sig_x_hat = self.backend.cat(
                    [path_class.signature(0, i).unsqueeze(1) for i in range(2, self.paths.shape[1])],
                    dim=1
                )
        elif self.method == 'numpy':
            def compute_signature_for_path(path, order):
                path_signatures = []
                for i in range(2, path.shape[0]):
                    subpath = path[:i, :]
                    # signature = self.iisignature.sig(subpath, order)
                    signature = np.r_[1., self.iisignature.sig(subpath, order)] # scalar_term=True
                    path_signatures.append(signature)
                return np.array(path_signatures)

            iterator = tqdm(self.paths, desc="Computing signatures") if verbose else self.paths
            self.sig_x_hat = Parallel(n_jobs=n_jobs)(delayed(compute_signature_for_path)(path, order) for path in iterator)
            self.sig_x_hat = np.array(self.sig_x_hat)


    def normalize_paths(self, method='logarithm'):
        """
        Normalising input paths
        -> for exponential-type behaviour (raw asset prices, GBM) it is best to perform
            the logarithm first, for the same purpose that log-returns are easier to work
            with (normality, stationarity, etc.). We also have much easier paths to work with
            in the signature setting due to using .diff() functionality.
        """
        if method == 'logarithm':
            self.paths = self.backend.log(self.paths + 1)
        elif method == 'scaling':
            self.paths /= self.paths[:, 0:1, :]

    def hoff_lead_lag(self):
        """
        Apply the Hoff lead-lag transformation to the paths, necessary for fitting.
        """
        if self.paths.shape[-1] == 2*(self.d + self.hidden_dim + 1):
            print("Hoff lead-lag already applied")
            return
        elif self.method == 'pytorch':
            self.paths = self._hoff_lead_lag_pytorch(self.paths)
        elif self.method == 'numpy':
            self.paths = self._hoff_lead_lag_numpy(self.paths)

    def _hoff_lead_lag_pytorch(self, x):
        x_rep = self.backend.repeat_interleave(x, repeats=4, dim=1)
        return self.backend.cat([x_rep[:, :-5], x_rep[:, 5:, :]], dim=2)

    def _hoff_lead_lag_numpy(self, x):
        x_rep = self.backend.repeat(x, repeats=4, axis=1)
        return self.backend.concatenate([x_rep[:, :-5], x_rep[:, 5:, :]], axis=2)

    def to_device(self, device):
        """
        Allows for cuda computation under pytorch backend.
        """
        if self.method == 'pytorch':
            self.paths = self.paths.to(device)
            self.device = device

    def pre_fit(self, order, n_jobs=-1, verbose=False):
        """
        Necessary for before being inputted into Sig-Trader class for fitting
        -> ensures we have the expected lead-lag signature computed
        """
        # Check if time is added, if not, add time
        if self.paths.shape[-1] == self.d + self.hidden_dim:
            self.add_time()

        if self.paths.shape[-1] == self.d + self.hidden_dim + 1:
            self.hoff_lead_lag()
        else:
            raise ValueError("Paths are not correctly formatted.")
        
        # Compute expected signature
        self.exp_sig_x_hat_ll = self.compute_expected_signature(order, n_jobs, verbose)
        self.sig_x_hat_ll = self.copy_(self.signatures)

    def pre_pnl(self, order, n_jobs=-1, verbose=False):
        """
        Necessary for before being inputted into Sig-Trader class for PnL computation
        -> ensures we have the signature computed for all paths
        """
        # Check if time is added, if not, add time
        if self.paths.shape[-1] == self.d + self.hidden_dim:
            self.add_time()
        if self.paths.shape[-1] != self.d + self.hidden_dim + 1:
            raise ValueError("Paths are not correctly formatted.")
        
        # Compute expected signature
        self.compute_signature_paths(order, n_jobs, verbose)