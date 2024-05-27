import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from esig import tosig
import itertools
import cvxpy as cp
from optimal_execution.shuffle import shuffle
from optimal_execution.utils import transform, find_dim, get_words
import iisignature

class SigExecution:
    def __init__(self,ES, q0=1, Lambda=1e-3, k=0, phi=0, alpha=10, N=2):
        """
        Optimal Execution for Signature Trading Strategies
        ----------
        This is based on the model presented in the paper:
        Kalsi, J., Lyons, T. and Perez Arribas, I., 2020. 
        Optimal execution with rough path signatures. 
        SIAM Journal on Financial Mathematics, 11(2), pp.470-493.

        The bulk of the code is based on the implementation by Imanol Perez:
        GitHub: https://github.com/imanolperez/optimal-execution-signatures

        Functionality:
        ----------
            - Given a batch of sample paths, fit the optimal execution Sig-Trading speed
            - Given an optimal strategy (linear functional) and a test path(s), compute the PnL

        Methods:
        ----------
        build_problem:
            given an expected signature, build and output the matrices/vectors required to solve the objective function
        fit:
            given the matrices/vectors from build_problem, solve the optimisation problem to find the optimal linear functional
        sig_speed:
            to output an operator that will calculate sig_speed of a given path
        get_analytics:
            for a given batch of sample paths, return the trading speed, inventory, and wealth through time
            
        Parameters
        ----------
        ES: np.array
            The expected signature of the midprice process
        q0: float
            initial inventory
        Lambda: float
            linear teporary market impact parameter
        k: float
            permanent market impact parameter
        phi: float
            running inventory penalty parameter (the larger phi, the greater the penalty)
        alpha: float
            terminal inventory parameter (the larger the alpha, the greater the inventory penalty)
        N: int
            order of truncation of the signature (and hence l)
        """
        
        self.ES, self.q0, self.Lambda, self.k, self.phi, self.alpha, self.N = ES, q0, Lambda, k, phi, alpha, N
        
        # order of signature required after shuffles and concatenation
        self.order = 2 * (self.N + 1) + 1
        self.built = False
        self.fitted = False

    def build_problem(self, phi=None, Lambda=None, q0=None, alpha=None, k=None, verbose=True, **kwargs):
        """
        Initialise the optimisation problem
        - Compute the words/multi-indices in the tensor algebra that correspond to the objective function
        - We will apply a linear functional to these multi-indices later, which is fitted in the "fit" method
        """
        # Housekeeping
        Lambda = Lambda or self.Lambda
        phi = phi or self.phi
        q0 = q0 or self.q0
        alpha = alpha or self.alpha
        k = k or self.k

        self.dim_l = find_dim(self.N)
        A = np.zeros((self.dim_l, self.dim_l))
        b = np.zeros(self.dim_l)
        c = 0.
        keys = get_words(2, self.order)

        # Define 1st set of words for shuffling
        alphabet = (0, 1)
        words = itertools.chain(*[itertools.product(alphabet, repeat=i) for i in range(self.N + 1)])

        # CREATE WORDS / MULTI INDICES FOR OBJECTIVE FUNCTION
        total = find_dim(self.N)
        iterator = tqdm(words, total=total) if verbose else words
    
        for w in iterator:
        
            w_idx = keys.index(w)
            
            # Define 2nd set of words for shuffling
            words2 = itertools.chain(*[itertools.product(alphabet, repeat=i) for i in range(self.N + 1)])

            # CREATE MATRIX INPUT FOR WORDS w, v
            # NB: We need to define a matrix for all shuffle products and combinations of w, v
            # These can be thought of as the cross terms
            
            for v in words2:
                v_idx = keys.index(v)

                # Shuffle Products
                w_shuffle_v = shuffle(w, v) # w \shuffle v
                w_shuffle_v_0 = [tuple(list(tau) + [0]) for tau in w_shuffle_v] # ( w \shuffle v ) 0
                w0_shuffle_v = shuffle(tuple(list(w) + [0]), v) # w0 \shuffle v
                w0_shuffle_v0 = shuffle(tuple(list(w) + [0]), tuple(list(v) + [0])) # w0 \shuffle v0
                w0_shuffle_v0_0 = [tuple(list(tau) + [0]) for tau in w0_shuffle_v0] #  ( w0 \shuffle v0 ) 0 
                
                # Associated Expected Signature Terms
                ES_w_shuffle_v_0 = sum(self.ES[tau] for tau in w_shuffle_v_0)
                ES_w0_shuffle_v = sum(self.ES[tau] for tau in w0_shuffle_v)
                ES_w0_shuffle_v0 = sum(self.ES[tau] for tau in w0_shuffle_v0)
                ES_w0_shuffle_v0_0 = sum(self.ES[tau] for tau in w0_shuffle_v0_0)
                                
                # Define the matrix-valued part of the objective function
                A[w_idx, v_idx] = -Lambda * ES_w_shuffle_v_0 + (k - alpha) * ES_w0_shuffle_v0 - (phi + k) * ES_w0_shuffle_v0_0

            # current expected sig term
            ES_w = self.ES[w]

            # CREATE VECTOR INPUT OBJECTS FOR WORDS w
            # NB: We need to create a vector for all single-word elements of the objective function
            w_shuffle_1 = shuffle(w, (1,)) # w \shuffle 1
            w_shuffle_10 = [tuple(list(tau) + [0]) for tau in w_shuffle_1] # ( w \shuffle 1 ) 0
            w0_shuffle_1 = shuffle(tuple(list(w) + [0]), (1,)) # (w0 \shuffle 1)
            
            w0 = tuple(list(w) + [0]) # w0
            w00 = tuple(list(w0) + [0]) # w00

            # Associated Expected Signature Terms
            ES_w_shuffle_10 = sum(self.ES[tau] for tau in w_shuffle_10)
            ES_w0_shuffle_1 = sum(self.ES[tau] for tau in w0_shuffle_1)
            ES_w0 = self.ES[w0]
            ES_w00 = self.ES[w00]

            # Define the vector-valued part of the objective function
            b[w_idx] = ES_w_shuffle_10 - ES_w0_shuffle_1 + (2 * alpha * q0 - q0 * k) * ES_w0 + 2 * phi * ES_w00

        # CREATE SCALAR INPUT OBJECTS FOR OBJECTIVE FUNCTION
        c = self.q0 * (self.ES[(1,)] + 1.) - alpha * q0 - q0 * phi * self.ES[(1,)]

        self.A, self.b, self.c = A,b,c
        self.built = True

    def fit(self, phi=None, Lambda=None, q0=None, alpha=None, k=None, verbose=False, **kwargs):
        """
        solve the optimisation problem to find the optimal linear functional l
        """
        # If any of the parameters are passed, rebuild the problem
        # If a parameter is not passed, the default self.param will be taken
        if any(x is not None for x in [phi, Lambda, q0, alpha, k]):
            self.build_problem(phi, Lambda, q0, alpha, k, verbose=verbose,**kwargs)

        if not self.built:
            self.build_problem(verbose=verbose)

        # Ensuring A is symmetric
        A = (self.A + self.A.T) / 2
        
        # Define objective (maximisation)
        l = cp.Variable(self.dim_l)
        objective = cp.Maximize(cp.quad_form(l, A) + self.b @ l)
        problem = cp.Problem(objective)

        # Solve objective for l
        problem.solve()
        
        self.l = l

        self.fitted = True

    def sig_speed(self, l=None, N=None, verbose=False):
        """
        to output an operator that will calculate sig_speed of a given path
        """
        # If no linear functional is passed or exists, then fit
        if l is None and not self.fitted:
            self.fit(verbose=verbose)
        # If no linear functional or truncation passed, take the previously fitted values
        if l is None:
            l = self.l.value
        if N is None:
            N = self.N
        
        # Define the sig_speed operator
        def f(path):
            path = transform(path)
            if N == 0:
                sig = np.array([1.])
            elif N == 1:
                sig = np.array([1., path[-1, 0] - path[0, 0], path[-1, 1] - path[0, 1]])
            else:
                sig = np.r_[1., iisignature.sig(path, N)]

            return sig.dot(l)
        
        return f
    
    def get_analytics(self, paths, speed=None, l=None, Lambda=None, q0=None, alpha=None, k=None, phi=None, verbose=True, **kwargs):
        """
        For a given batch of sample paths [num_paths, path_length, dim] 
        return
            - the trading speed through time
            - inventory through time
            - wealth through time
        """
        # Housekeeping
        if phi is not None:
            self.fit(phi, verbose=verbose)
            l = self.l.value
        if l is None:
            if self.fitted:
                l = self.l.value
            else:
                self.fit(verbose=verbose)
                l = self.l.value
        speed = speed or self.sig_speed(l=l, verbose=verbose)
        Lambda = Lambda or self.Lambda
        q0 = q0 or self.q0
        alpha = alpha or self.alpha
        k = k or self.k
        
        paths_Qt = []
        paths_wealth = []
        paths_speed = []

        # Loop through sample paths
        iterator = tqdm(paths) if verbose else paths
        for path in iterator:
            WT = 0.
            pnl = [0.]
            Qt = [0.]
            speeds = []
            permanent_impact = 0.
            for i in range(len(path) - 1):
                delta_t = path[i + 1, 0] - path[i, 0] # time is the 0th dimension
                speed_t = speed(np.array(path[:i + 1]))

                # permanent impact = k * int_t_0 theta_s ds
                permanent_impact += k * speed_t * delta_t 
                # temporary impact = \lambda theta_s
                temporary_impact = Lambda * speed_t

                # Wealth process = (midprice - impact) * dQ
                WT += (path[i, 1] - permanent_impact - temporary_impact) * speed_t * delta_t
                pnl.append(WT)
                Qt.append(Qt[-1] + speed_t * delta_t)
                speeds.append(speed_t)
            
            Qt = q0 - np.array(Qt)
            WT += Qt[-1] * (path[-1, 1] - permanent_impact  - alpha * Qt[-1])
            pnl.append(WT)
            paths_Qt.append(Qt)
            paths_wealth.append(pnl)
            paths_speed.append(speeds)

            
        return paths_speed, paths_Qt, paths_wealth