import torch
import signatory
import itertools
from typing import List, Tuple
from mean_variance.utils import sigwords, word_to_loc, shuffle

def shuffle_prod_of_two_vectors(v, w, dim_v, dim_w, order_v, order_w, scalar_term: bool = False):
    """
    Compute the shuffle product of
        $v \sha w $,
    where $v, w \in T((\mathbb{R}^d))$.
    """
    sig_dim: int  = signatory.signature_channels(max(dim_v, dim_w), order_v + order_w, scalar_term=scalar_term)

    # create the words you want to shuffle
    words_v = sigwords(order_v, dim_v, scalar_term=scalar_term)
    words_w = sigwords(order_w, dim_w, scalar_term=scalar_term)

    e_vecs = list()
    # compute shuffle product
    for word_v, word_w in itertools.product(words_v, words_w):
        loc_v = word_to_loc([word_v], dim_v, scalar_term=scalar_term)
        loc_w = word_to_loc([word_w], dim_w, scalar_term=scalar_term)

        loc_v_shuffle_w = word_to_loc(shuffle(word_v, word_w), max(dim_v, dim_w), scalar_term=scalar_term)
        e = torch.zeros(1, sig_dim).float()
        vw = v[:, loc_v] * w[:, loc_w]
        for loc in loc_v_shuffle_w:
            e[:, loc] = e[:, loc] + vw
        e_vecs.append(e)
    v_shuffle_w = torch.cat(e_vecs, dim=0).sum(0, keepdims=True)
    
    return v_shuffle_w


def precompute_shuffle_locs(dim_v, dim_w, order_v, order_w, scalar_term: bool = True):
    """
    To store pre-computed shuffles if needed to speed up online computation
    """
    locs_v = list()
    locs_w = list()
    locs_vsw = list()

    words_v = sigwords(order_v, dim_v, scalar_term=scalar_term)
    words_w = sigwords(order_w, dim_w, scalar_term=scalar_term)
    for word_v, word_w in itertools.product(words_v, words_w):
        locs_v.append(word_to_loc([word_v], dim_v, scalar_term=scalar_term))
        locs_w.append(word_to_loc([word_w], dim_w, scalar_term=scalar_term))
        locs_vsw.append(word_to_loc(shuffle(word_v, word_w), max(dim_v, dim_w), scalar_term=scalar_term))

    sig_dim = signatory.signature_channels(max(dim_v, dim_w), order_v + order_w, scalar_term=scalar_term)

    indices = list()
    for i, locs in enumerate(locs_vsw):
        indices = indices + [(i*sig_dim)+loc for loc in locs]
    return locs_v, locs_w, locs_vsw, indices


def shuffle_prod_efficient(v, w, locs_v, locs_w, locs_vsw, indices, dim_w, dim_v, order_w, order_v, scalar_term: bool = True):
    """
    Sped up version of shuffle product for vectors
    """
    sig_dim: int  = signatory.signature_channels(max(dim_v, dim_w), order_v + order_w, scalar_term=scalar_term)
    vw = (v[0, locs_v] * w[0, locs_w])[:, 0]
    e = torch.zeros(len(locs_v)*sig_dim).float().to(v.device)

    vw_list = list()
    for i, (vw_i, locs) in enumerate(zip(vw, locs_vsw)):
        vw_list.append(vw_i.repeat(len(locs)))
    vw = torch.cat(vw_list)

    indices = torch.tensor(indices).to(v.device)
    e = e.index_add(0, indices, vw)
    e = e.reshape(len(locs_v), sig_dim)
    vsw = e.sum(0, keepdims=True)
    return vsw