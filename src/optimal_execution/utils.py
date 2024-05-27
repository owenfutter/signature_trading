import numpy as np
from tqdm.auto import tqdm
import iisignature
from esig import tosig

def transform(X):
    if len(X) == 1:
        return np.array([[-X[0, 0], X[0, 1]], [X[0, 0], X[0, 1]]])
    new_X = [[-X[1, 0], X[0, 1]]]
    for x_past, x_future in zip(X[:-1], X[1:]):
        new_X.append(x_past)
        new_X.append([x_past[0], x_future[1]])
        
    new_X.append(X[-1])
    
    return np.array(new_X)

def find_dim(N):
    """
    Find the dimension of the signature space for a given order N.
    d = \sum_{i=1}^N 2^i
    """
    if N == 0:
        return 0
    if N == 1:
        return 3
    
    return tosig.sigdim(2, N)

def get_words(dim, order):
    """
    Return all words (multi-indices) of a given dimension and order in the tensor algebra.
    """
    keys = [tuple([t]) if isinstance(t, int) else t for t in map(eval, tosig.sigkeys(dim, order).split())]
    keys = [tuple(np.array(t) - 1) for t in keys]
    return keys

def Cost(path, speed, q0, Lambda, k, phi, alpha, **kwargs):
    WT = 0.
    QT = q0
    L2_penalty = 0.
    delta_t = path[1, 0] - path[0, 0]
    permanent_impact = 0.
    for i in range(len(path)):
        speed_t = speed(np.array(path[:i + 1]))
        permanent_impact += k * speed_t * delta_t
        
        temporary_impact = Lambda * speed_t
        
        WT += (path[i, 1] - permanent_impact - temporary_impact) * speed_t * delta_t
        QT -= speed_t * delta_t
        L2_penalty += QT**2 * delta_t

    C = WT + QT * (path[-1, 1] - permanent_impact - alpha * QT) - phi * L2_penalty
    
    return C