import anndata as ad
import math
import operator
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn as sk
import toolz
import typing
import warnings
from importlib import reload
from scipy import sparse
from scipy import stats
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from toolz import curry
from umap import UMAP
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform, cdist
from scipy import linalg
from typing import Optional, Iterable, Tuple, Union
from collections import Counter
from collections import defaultdict, OrderedDict
import scipy

def cyclicPermutation(n :int, k=None):
    if k is None:
        k = np.random.randint(0,n)
    ind = np.mod(np.arange(k, n+k, 1), n)
    return ind

@curry
def generateMRNA(a, b, g, dt, s0, u0, n, zinf = 0, dif=0, rate_distort=True):
    s = np.ones(n) * s0
    u = np.ones(n) * u0
    v = 1.0
    vv = 1.0
    if not isinstance(a, np.ndarray):
        a = a * np.ones(n-1)
    if not isinstance(b, np.ndarray):
        b = b * np.ones(n-1)
    if not isinstance(g, np.ndarray):
        g = g * np.ones(n-1)
    for i in range(n-1):
        c = np.random.rand() > zinf
        if rate_distort:
            v = 1 + 0.25 * np.random.randn()
            v = np.clip(v, 0, None)
            vv = 1 + 0.25 * np.random.randn()
            vv = np.clip(vv, 0, None)
        u[i + 1] = u[i] +  dt * (c * a[i] - v*b[i] * u[i])
        s[i+1] = s[i] + dt * (v*b[i] * u[i] - vv*g[i] * s[i]) + np.random.randn() * dif * dt
        s[i+1] = np.clip(s[i+1], 0, None)
        u[i+1] = np.clip(u[i+1], 0, None)
    r = s + u
    return s, r, u

def maskedScale(X, mask, axis=0, replacewithmin = False,):
    Y = np.where(mask, np.nan, X)
    s = np.nanstd(Y, axis=axis, keepdims=True)
    m = np.nanmean(Y, axis=axis, keepdims=True)
    Y = (Y - m) / s
    if replacewithmin:
        rval = np.nanmin(Y)
        Y = np.where(np.isnan(Y), rval, Y)
    return Y

def simulateGenes(n_genes : int, a, b, g, dt, s0, u0, n, zinf=0, dif=0, rscale=False, n_repeats=5, rate_distort=True):
    S = np.zeros((n * n_repeats, n_genes))
    R = np.zeros((n * n_repeats, n_genes))
    t = np.zeros((n * n_repeats, ))
    for i in range(n_genes):
        l = cyclicPermutation(len(a), )
        if rscale:
            scale = 0.6 + np.random.rand()*3
        alpha = a if np.ndim(a) < 2 else a[:,i]
        beta = b if np.ndim(b) < 2 else b[:,i]
        gamma = g if np.ndim(g) < 2 else g[:,i]
        for j in range(n_repeats):
            s, r, u = generateMRNA(alpha[l],beta,gamma,dt,0,0, n, zinf=zinf, dif=dif, rate_distort=rate_distort)
            if rscale:
                s = s*scale
                r = r*scale
            S[j*n:(j+1)*n,i] = s
            R[j*n:(j+1)*n,i] = r
            t[j*n:(j+1)*n,] = np.linspace(0, dt*(n-1), n)
    temp = (1 + np.random.randn(*S.shape) * 0.15).clip(0.55)
    S = S * temp
    R = R * (1 + np.random.randn(*S.shape).clip(0) * 0.15) * temp
    adata = ad.AnnData(
            X = S.copy(),
            layers = {
                "spliced" : S.copy(),
                "total" : R.copy(),
                "unspliced" : R - S,
                },
            obs = {
                "time" : t,
                #"time" : np.cumsum(np.repeat((0,dt), (1,n-1))),
                },
            )
    return adata

