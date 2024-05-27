import torch
import itertools
from typing import List, Tuple


def to_numpy(x: torch.Tensor):
    """
    Casts torch.Tensor to a numpy ndarray.
    """
    return x.detach().cpu().numpy()

def add_time(x: torch.Tensor) -> torch.Tensor:
    """ 
    Time-augmentation of a path. \hat S_t = (t, S_t). 
    This is for equal spaced time series paths
    """
    size, length, dim = x.shape
    time_vector = torch.linspace(0, 1, length).reshape(1, -1, 1).repeat(size, 1, 1).to(x.device)
    x_hat = torch.cat([time_vector, x], dim=2) # ensures that time is the zero-th index

    return x_hat

def add_power_time(x: torch.Tensor, power=1) -> torch.Tensor:
    """ 
    Time-augmentation of a path. \hat S_t = (t, S_t), for when time is non-linear (polynomial)
    """
    # Time will always be between 0 and 1
    size, length, dim = x.shape
    time_vector = torch.linspace(0, 1, length)**power # polynomial time
    time_vector = time_vector.reshape(1, -1, 1).repeat(size, 1, 1).to(x.device)
    x_hat = torch.cat([time_vector, x], dim=2) # ensures that time is the zero-th index

    return x_hat

def add_exponential_time(x: torch.Tensor, lam=1) -> torch.Tensor:
    """ 
    Time-augmentation of a path. \hat S_t = (t, S_t), for exponential time
    """
    # Time will always be between 0 and 1
    def exp_transform(t, lam):
        return (torch.exp(torch.tensor(lam))-torch.exp(lam*(1-t)))/(torch.exp(torch.tensor(lam))-1)
    size, length, dim = x.shape
    time_vector = exp_transform(torch.linspace(0, 1, length), lam)
    time_vector = time_vector.reshape(1, -1, 1).repeat(size, 1, 1).to(x.device)
    x_hat = torch.cat([time_vector, x], dim=2) # ensures that time is the zero-th index

    return x_hat

def hoff_lead_lag(x:torch.Tensor) -> torch.Tensor:
    """
    Hoff Lead Lag Transform
    """
    x_rep = torch.repeat_interleave(x, repeats=4, dim=1) 
    # slightly different version of the lead-lag transform above (larger lag shift)
    return torch.cat([x_rep[:, :-5], x_rep[:, 5:, :]], dim=2)

def batches_to_path(batches):
    """
    input: in batches of size [num_batches, path_length, dim]
    
    output: one continuous path of size [num_batches*(path_length-1), dim]
    """

    return torch.cat([torch.ones(1,1), torch.diff(batches,dim=1).flatten(end_dim=1).cumsum(0)+1],dim=0)

def sigwords(order: int, dim: int, scalar_term:bool = False) -> torch.Tensor:
    """
    For a given order N and dimension d, we obtain the words in the tensor algebra T^N((R^d))
    """
    words = []
    if scalar_term:
        words.append(())
    for n in range(1, order + 1):
        for w in itertools.product(*[list(range(dim)) for _ in range(n)]):
            words.append(w)
    return words

def sigwords_from_index_list(order, indices):
    """
    Given some indices in a vector, and the given order, obtain the corresponding words
    """
    words = []
    for n in range(1, order + 1): 
        for w in itertools.product(*[indices for _ in range(n)]): 
            words.append(w)
    return words 

def word_to_loc(words: List[Tuple[int]], dim: int, scalar_term: bool=False) -> List[int]:
    """ Maps words of tensor algebra to locations in the 'flattened' vector space. """
    words = list(words)
    loc = []
    for i in words:
        if len(i)>0:
            assert max(i) <= (dim-1), 'Words must be <= dimension-1. Received word %s and dimension %s' % (i, dim)
        n = len(i)
        if dim > 1:
            index = (dim ** n - 1) / (dim - 1) - 1 + sum(
                torch.IntTensor(i) * torch.IntTensor([dim ** (n - j) for j in range(1, n + 1)]))
        else:
            index = len(i) - 1
        if scalar_term:
            index = index + 1
        loc.append(int(index))
    return loc

def concatenate(u: int, words: List[Tuple[int]]):
    """ Concatenates a letter with each word in an interable. """
    for word in words:
        yield tuple([u] + list(word))

def shuffle(word_1: Tuple[int], word_2: Tuple[int]):
    """Computes the shuffle product of two words."""
    if len(word_1) == 0:
        return [word_2]
    if len(word_2) == 0:
        return [word_1]
    gen1 = concatenate(word_1[0], shuffle(word_1[1:], word_2))
    gen2 = concatenate(word_2[0], shuffle(word_1, word_2[1:]))
    return itertools.chain(gen1, gen2)

def sig_dim(dim, order):
    if dim == 0:
        raise ValueError("dim must be > 0")
    if dim == 1:
        return order + 1
    if order == 0:
        return 1
    else:
        return int((dim**(order + 1) - 1) / (dim - 1))