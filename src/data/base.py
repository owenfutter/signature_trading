import numpy as np
from tqdm.auto import tqdm
import iisignature
from joblib import Parallel, delayed

import optimal_execution.utils as utils
import optimal_execution.tensor_algebra as ta


class Midprice(object):
    """
    Base class for data (i.e. midprice models).

    Functionality:
    ----------
        - To be passed into a model (e.g Brownian Motion)
        - To construct sample paths and the corresponding expected signature
    
    Methods:
    ----------
    build: 
        Creates paths and expected Signature

    Parameters
    ----------
    null
    -> to be passed into model object
    """
    def __init__(self, *args, **kwargs):
        pass
   
    @staticmethod
    def _sig(path, order):
        return np.r_[1., iisignature.sig(utils.transform(path), order)]

    def _generate(self, seed):
        np.random.seed(seed)

        return self.generate()

    def generate(self):
        """
        Generate a sample path
        """

        raise NotImplementedError("Generator not implemented")

    def build(self, n_paths=1000, order=6, verbose=False):
        """
        Compute paths and expected signature
        """

        # Create paths
        iterator = tqdm(range(n_paths), desc="Building paths") if verbose else range(n_paths)
        
        paths = Parallel(n_jobs=-1)(delayed(self._generate)(seed) \
                                    for seed in iterator)


        iterator = tqdm(paths, desc="Computing signatures") if verbose else paths
        # Compute signatures
        sigs = Parallel(n_jobs=-1)(delayed(self._sig)(path, order) \
                                   for path in iterator)

        # Compute Expected Signature, a Tensor object
        ES = ta.Tensor(np.mean(sigs, axis=0), 2, order)

        return np.array(paths), ES
