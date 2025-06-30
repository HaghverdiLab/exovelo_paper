import anndata as ad
import copy
import functools
import math
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import operator
import pandas as pd
import polars as pl
import polars.selectors as cs
import scanpy as sc
import seaborn as sns
import sklearn as sk
import toolz
import typing
import warnings
from dataclasses import dataclass
from importlib import reload
from matplotlib.backends.backend_pdf import PdfPages
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
from typing import Optional, Iterable, Tuple, Union, List
from collections import Counter
import scipy
from collections import defaultdict, OrderedDict
from scipy.optimize import minimize

@curry
def shiftHexStr(s, base=16, shift = (2*16**2 + 2*16 + 2), k=0):
    """
    s : string representing a hex
    k: remove the first k chars (if they are an indicating prefix)
    returns hex
    base: 16 or any other valid base matching whatever the string represents
    shift: value to shift the original hex with
    """
    prefix = s[:k]
    n = int(s[k:], base=base)
    m = n + shift
    r = prefix + hex(m)[2:]
    return r

def relativeResidualVariance(X_in, X_out, ):
    """
    returns var(X_in - X_out) / var(X_in)
    where var(x) := trace(x.T @ x)
    expects 2-d array type as inputs
    """
    vr = np.trace( (X_in - X_out).T @ (X_in - X_out) )
    vx = np.trace( X_in.T @ X_in )
    return vr / vx

def relativeError(X_in, X_out, axis=-1, ord=2, adjust : bool=True,):
    """
    compute relative error between X_in and X_out defined as:
    `|X_in - X_out| / |X_in|`
    Reduces to the mean over the rest of the axes
    parameter:
    X_in, X_out : array types
    ord, axis: same as in np.linalg.norm
    adj: `bool (True)`: if True 1 is added to |X_in| b/c relative error of null input is not informative.
    """
    eps = 1e-12
    result = np.linalg.norm((X_in - X_out), ord=ord, axis=axis, keepdims=True,)
    temp = np.linalg.norm((np.abs(X_in) + eps), ord=ord,axis=axis, keepdims=True,)
    if adjust:
        temp += 1
    return (result / temp).mean()


def unsparse(x):
    """
    Converts sparse types to numpy array
    also converts numpy matrix type to array type.
    """
    if sparse.issparse(x):
        return x.toarray()
    elif type(x) == np.matrix:
        return x.A
    else:
        return x

def unsparseAnnData(
        adata : ad.AnnData,
        doX : bool = True,
        ):
    """
    Converts 'in place' all the layers of an andata object to numpy array
    using the 'unsparse' function.
    """
    if doX:
        adata.X = unsparse(adata.X)
    for layer in adata.layers:
        layer = str(layer)
        adata.layers[layer] = unsparse(adata.layers[layer])
    return


def normalize_batches(Xs: List[np.ndarray], ):
    """
    l1 normalize a list of cell×gene count matriced so that in the output
    each row equals the target_sum.
    target_sum is calculated as follows:
        take the median row_sum per from each matrix
        take the median of these medians.
    """
    batch_medians = [np.median(X.sum(1)) for X in Xs]
    target_sum = np.median(batch_medians)
    return [target_sum * normalize(
        X,
        "l1",
        axis=1,
    ) for X in Xs]

def l1Normalize(
        #data : typing.Union[np.ndarray, np.matrix, sparse.spmatrix, sparse.sparray,],
        data ,
        target_sum : float=1,
        axis : int = 1,
        ):
    """
    Perform l1 normalization on arrays or matrices.
    parameters
    ----------
    data: array or matrix type (sparse or dense)
    target_sum: the value that each 'row' along 'axis' will sum up to. default=1.
    axis: the axis to normalzie, defaults to 1 (rows will sum to target_sum)
    """
    d = {"axis" : axis }
    if not (sparse.issparse(data) or type(data) == np.matrix):
        d['keepdims']=True
    return unsparse(target_sum * data / (np.abs(data)).sum(**d))
#    if sparse.issparse(data) or type(data) == np.matrix:
#        return target_sum * data / np.abs(data).sum(axis=axis)
#    else:
#        return target_sum * data / np.abs(data).sum(axis=axis, keepdims=True)

def normalizeCombined(
        adata,
        xlayer : str,
        ylayer : str,
        clayer : str,
        suffix : str="_n",
        o = 2, #order
        target_sum : float=1e4,
        ):
    rowsum = np.linalg.norm(adata.layers[clayer] / target_sum, ord=o, axis=1,).reshape(-1,1)
    adata.layers[clayer + suffix] = adata.layers[clayer] / rowsum
    adata.layers[xlayer + suffix] = adata.layers[xlayer] / rowsum
    adata.layers[ylayer + suffix] = adata.layers[ylayer] / rowsum


def l1NormalizeCombine(
        adata : ad.AnnData,
        xlayer : str,
        ylayer : str,
        combined : bool=True,
        suffix : str="_n",
        target_sum : float=1,
        axis : int = 1,
        ):
    """
    Perform l1 normalization on two layers of an annotated data object,
    prsetving  relative size to each other.
    parameters
    ----------
    adata: AnnData object
    xlayer: `str` indicates first layer
    ylayer: `str` indicates second layer
    combined: If True preserves relative sum. Otherwise each is normalized separately.
    suffix: `str`: return new layer named `old_layer_suffix`
    target_sum: the value that each 'row' along 'axis' will sum up to. default=1.
    axis: the axis to normalzie, defaults to 1 (rows will sum to target_sum)
    """
    d = {"axis" : axis}
    if not (sparse.issparse(adata.layers[ylayer]) or type(adata.layers[ylayer]) == np.matrix):
        d['keepdims']=True
    if combined:
        rsum = (np.abs(adata.layers[xlayer]) + np.abs(adata.layers[ylayer])).sum(**d)
        adata.layers[xlayer + suffix] = unsparse(target_sum * adata.layers[xlayer] / rsum)
        adata.layers[ylayer + suffix] = unsparse(target_sum * adata.layers[ylayer] / rsum)
    else:
        adata.layers[xlayer + suffix] = unsparse(target_sum * adata.layers[xlayer] / np.abs(adata.layers[xlayer]).sum(**d) )
        adata.layers[ylayer + suffix] = unsparse(target_sum * adata.layers[ylayer] / np.abs(adata.layers[ylayer]).sum(**d) )

def regressGamma(
        adata : ad.AnnData,
        target_layer: str='old_n',
        tkey : str='time',
        n_jobs : int=8,
        #fit_intercept : bool=True,
        ):
    """
    Estimate degradation rate from 'old' rna slamseq data in a kinetic pulse experiment
    or from 'old new' rna from chase experiment.
    The assumption is that the distribution of rna counts doesn't change w.r.t pulse time.
    `\log(\mean(O(t))) = log(mean(O(0))) - \gamma t`
    parameters:
    `adata` : anndata object,
    `target_layer' : str. the key to the layer to be used as the old rna data.
    `tkeye` : str. the key in adata.obs for the time parameter
    `n_jobs` :  number of threads
    returns a polars dataframe with the results
    """
    tvals = np.unique(adata.obs[tkey]).tolist()
    ns = []
    for t in tvals:
        ns.append(
                unsparse(adata.layers[target_layer])[adata.obs[tkey] == t,:].mean(0))
    ns = np.array(ns)
    ts = np.array(tvals)
    logs = np.log(1e-16 + ns).T
    result = np.zeros((adata.n_vars, 5),)
    for g in range(adata.n_vars):
        res = stats.linregress(-ts, logs[g])
        result[g][:5] = np.array(res[:])
        #gamma = np.abs(res[0])
        #ks = (1 - np.exp( - gamma * ts)) / gamma
    df = pl.DataFrame(
            data = result,
            schema = ['g_slope', 'g_intercept', 'g_rvalue', 'g_pvalue', 'g_stderr', ],
            )
    #adata.var[['g_slope', 'g_intercept', 'g_rvalue', 'g_pvalue', 'g_stderr', ]] = result
    return df

def regressGammaQuant(
        adata : ad.AnnData,
        target_layer: str='old_n',
        tkey : str='time',
        n_jobs : int=8,
        tmin : typing.Optional[float] = None,
        q : typing.Optional[float] = 0.7,
        control_layer : str ='new_n',
        return_means : bool=False,
        logmean : bool=True,
        #fit_intercept : bool=True,
        ):
    """
    Estimate degradation rate from 'old' rna slamseq data in a kinetic pulse experiment
    or from 'old new' rna from chase experiment.
    The assumption is that the distribution of rna counts doesn't change w.r.t pulse time.
    `\log(\mean(O(t))) = log(mean(O(0))) - \gamma t`
    parameters:
    `adata` : anndata object,
    `target_layer' : str. the key to the layer to be used as the old rna data.
    `tkeye` : str. the key in adata.obs for the time parameter
    `n_jobs` :  number of threads
    `tmin`: optinal float. If set, only measurements with labeling time longer than tmin will be used.
    `q`: optional float. if not none only the values that are in the upper than q quantile will be used for the regression.
        perhaps this may help in zero-inflation correction.
    returns a polars dataframe with the results
    """
    tvals = np.unique(adata.obs[tkey]).tolist()
    if tmin:
        tvals = [t for t in tvals if t > tmin]
    ts = np.array(tvals)
    result = np.zeros((adata.n_vars, 5),) + 100
    result[:,2] = -100
    lmeans = np.zeros((adata.n_vars, len(tvals)))
    for g in range(adata.n_vars):
        ys = []
        for t in tvals:
            temp  = unsparse(adata.layers[target_layer])[adata.obs[tkey] == t,g]
            #tempctrl  = unsparse(adata.layers[control_layer])[adata.obs[tkey] == t,g]
            #if (tempctrl > 0).sum() == 0:
            #    continue
            #temp = temp[tempctrl > 0]
            #yq = np.quantile(temp, q)
            #temp = temp[temp >= yq]
            temp = temp[temp > 0]
            if len(temp) == 0:
                continue
            #temp = np.log(temp).mean()
            if logmean:
                temp = np.log(temp.mean())
            else:
                temp = np.mean(np.log(temp))
            #temp = np.mean(np.log(temp + 1e-16))
            ys.append(temp)
        if len(ys) < len(ts):
            continue
        logs = np.array(ys)
        lmeans[g,:] = logs
        res = stats.linregress(-ts, logs)
        result[g][:5] = np.array(res[:])
    df = pl.DataFrame(
            data = result,
            schema = ['g_slope', 'g_intercept', 'g_rvalue', 'g_pvalue', 'g_stderr', ],
            )
    if return_means:
        return df, lmeans, ts
    return df



def regressKappaFromAdTest(
        adata,
        x_layer, # total
        y_layer, # old ~ total or new ~ total
        q : float = 0.9,
        tkey: str = 'time',
        method: str='otr', #'ntr'
        tmin : typing.Optional[float] = None,
        update_data : bool=True,
        ) -> pl.DataFrame:
    """
    uses steady state assumption to estimate kappa = 1 - exp(-gamma t) from kinetic RNA labeling experiment
    with method='ntr' regression is done using new ~ total
    with method='otr' regression is done on old ~ total
    parameters
    ----------
    adata : anndata object
    x_layer :str type indicating the total rna (the predictive variable)
    y_layer : str type idicating the dependent variable (new or old rna)
    q : float in range (0,1) to set the top quantile expression counts whose cells are assumed to be in steady state.
    tkey: str indicating the time column
    method: str either 'ntr' for new rna or 'otr' for old rna.
    tmin: non-negative float or None. If set, only measurement with labeling time longer than tmin will be used for the calculations.
        This is meant to filter out short labeling times which produce possibly unreliable data.
    update_data : bool; if True will write results into adata.var not functional currently
    """
    tvals = np.sort(np.unique(adata.obs[tkey])).tolist()
    if tmin:
        tvals = [t for t in tvals if t > tmin]
    result = np.zeros((adata.n_vars, 4),) - 1
    dfs = []
    for t in tvals:
        for g in range(adata.n_vars):
            x = adata[adata.obs[tkey] == t][:,g].layers[x_layer].flatten()
            y = adata[adata.obs[tkey] == t][:,g].layers[y_layer].flatten()
            ###
            x = x[y > 0]
            y = y[y > 0]
            if len(x) == 0:
                result[g][3] = 100
                #result[g][:3] = -100 * np.random.rand()
                result[g][:3] = -100
                continue
            # if o/r > 0.95 assume there is no transcription
            # if n/r > 0.95 assume it is in SS and gamma can't be obtained form the slope
            marksOn = (y / x < 0.95) * (y / x > 0.05) # mark if gene is on
            #marksOn = (y / x < 0.85) * (y / x > 0.15) # mark if gene is on
            if marksOn.sum() == 0:
                result[g][3] = 100
                #result[g][:3] = -100 * np.random.rand()
                result[g][:3] = -100
                #print(g,"0!")
                continue
            x = x[marksOn]
            y = y[marksOn]
            ###
            #yq = np.quantile(y/x, q)
            #markq = y/x >= yq
            xq = np.quantile(x, q)
            yq = np.quantile(y, q)
            markq = (x >= xq) * (y >= yq)
            if markq.sum() == 0:
                result[g][3] = 100
                #result[g][:3] = -100 * np.random.rand()
                result[g][:3] = -100
                #print(g,"0!!")
                continue
            x = x[markq].reshape(-1,1)
            y = y[markq].reshape(-1,1)
            ###
            reg = LinearRegression(fit_intercept=False,).fit(x,y)
            result[g][0] = kappa = reg.coef_[0,0]
            fpval = f_regression(x,y)[1][0]
            result[g][3] = fpval
            if method == 'otr':
                kappa = 1 - kappa
                result[g][0] = kappa
            result[g][1] = score = reg.score(x,y)
            result[g][2] = gamma = -np.log(1 - kappa) / t
            #result[g][:5] = np.array(res[:])
        df = pl.DataFrame(
                data = result,
                schema = ['k_'+str(t), 'score_' + str(t), 'gamma_' + str(t), 'pval_' + str(t), ],
                )
        dfs.append(df)
    dfs = pl.concat(dfs, how='horizontal')
    score_min = pl.Series(
            dfs.select(cs.matches("score")).to_numpy().min(1))
    dfs = dfs.with_columns(score_min.alias("score_min"))
    pval_max = pl.Series(
            dfs.select(cs.matches("pval")).to_numpy().max(1))
    dfs = dfs.with_columns(pval_max.alias("pval_max"))
    if len(tvals) > 2:
        dfs = regressGammaFromKappa(dfs, np.array(tvals).reshape(-1,1))
    return dfs

def regressGammaFromKappa(
        df,
        ts,
        ) -> pl.DataFrame:
    """
    help function to estimate gamma by regression on the estimated kappas
    only meant for internal use within a larger function.
    """
    n = len(df)
    ks = df.select(cs.matches("k_")).to_numpy() - 1e-9
    result = -np.ones((n,3))
    for g in range(n):
        if df[g,0] == -100:
            result[g,0] = 100
            result[g,1] = -100
            result[g,2] = 100
            continue
        y = np.log(1 - ks[g]).reshape(-1,1)
        reg = LinearRegression(fit_intercept=False,).fit(-ts,y)
        fpval = f_regression(-ts,y)[1][0]
        result[g,0] = gamma = reg.coef_.flatten()[0]
        result[g,1] = score = reg.score(-ts,y)
        result[g,2] = fpval
    return df.with_columns(
            pl.Series(result[:,0]).alias('gamma_reg'),
            pl.Series(result[:,1]).alias('score2'),
            pl.Series(result[:,2]).alias('pval2'),
            )

def regressKappaFromOld(
        adata,
        x_layer, # total
        y_layer, # old
        tkey : str = 'time',
        qx : float = 0.9,
        qy : float = 0.0,
        suffix : typing.Optional[str] = "x",
        ):
    """
    Estimate gamma from kinetic pulse experiment under steady state assumption:
    Old = Total * exp(-gamma t)
    """
    tvals = np.sort(np.unique(adata.obs[tkey])).tolist()
    result = np.zeros((adata.n_vars, 4),) - 1
    dfs = []
    for t in tvals:
        for g in range(adata.n_vars):
            x = adata[adata.obs[tkey] == t][:,g].layers[x_layer].flatten()
            y = adata[adata.obs[tkey] == t][:,g].layers[y_layer].flatten()
            x = x[y > 0]
            y = y[y > 0]
            if len(x) == 0:
                result[g][3] = 100
                #result[g][:3] = -100 * np.random.rand()
                result[g][:3] = -100
                continue
            xq = np.quantile(x, qx)
            yq = np.quantile(y, qy)
            mark = (x > xq) * (y > yq)
            x = x[mark].reshape(-1,1)
            y = y[mark].reshape(-1,1)
            if len(x) == 0:
                result[g][3] = 100
                #result[g][:3] = -100 * np.random.rand()
                result[g][:3] = -100
                continue
            reg = LinearRegression(fit_intercept=False,).fit(x,y)
            result[g][0] = kappa = 1 - reg.coef_[0,0]
            fpval = f_regression(x,y)[1][0]
            result[g][3] = fpval
            result[g][1] = score = reg.score(x,y)
            result[g][2] = gamma = -np.log(1 - kappa) / t
        df = pl.DataFrame(
                data = result,
                schema = ['k_'+str(t), 'score_' + str(t), 'gamma_' + str(t), 'pval_' + str(t), ],
                )
        dfs.append(df)
    dfs = pl.concat(dfs, how='horizontal')
    score_min = pl.Series(
            dfs.select(cs.matches("score")).to_numpy().min(1))
    dfs = dfs.with_columns(score_min.alias("score_min"))
    pval_max = pl.Series(
            dfs.select(cs.matches("pval")).to_numpy().max(1))
    dfs = dfs.with_columns(pval_max.alias("pval_max"))
    if len(tvals) > 2:
        dfs = regressGammaFromKappa(dfs, np.array(tvals).reshape(-1,1))
    return dfs

def regressKappaFromNew(
        adata,
        x_layer, # total
        y_layer, # new
        tkey : str = 'time',
        qx : float = 0.9,
        qy : float = 0.0,
        suffix : typing.Optional[str] = "x",
        ):
    """
    Estimate gamma from kinetic pulse experiment under steady state assumption:
    new = Total * (1 - exp(-gamma t) )
    """
    tvals = np.sort(np.unique(adata.obs[tkey])).tolist()
    result = np.zeros((adata.n_vars, 4),) - 1
    dfs = []
    for t in tvals:
        for g in range(adata.n_vars):
            x = adata[adata.obs[tkey] == t][:,g].layers[x_layer].flatten()
            y = adata[adata.obs[tkey] == t][:,g].layers[y_layer].flatten()
            x = x[y > 0]
            y = y[y > 0]
            if len(x) == 0:
                result[g][3] = 100
                #result[g][:3] = -100 * np.random.rand()
                result[g][:3] = -100
                continue
            xq = np.quantile(x, qx)
            yq = np.quantile(y, qy)
            mark = (x > xq) * (y > yq)
            x = x[mark].reshape(-1,1)
            y = y[mark].reshape(-1,1)
            if len(x) == 0:
                result[g][3] = 100
                #result[g][:3] = -100 * np.random.rand()
                result[g][:3] = -100
                continue
            reg = LinearRegression(fit_intercept=False,).fit(x,y)
            result[g][0] = kappa = reg.coef_[0,0]
            fpval = f_regression(x,y)[1][0]
            result[g][3] = fpval
            result[g][1] = score = reg.score(x,y)
            result[g][2] = gamma = -np.log(1 - kappa) / t
        df = pl.DataFrame(
                data = result,
                schema = ['k_'+str(t), 'score_' + str(t), 'gamma_' + str(t), 'pval_' + str(t), ],
                )
        dfs.append(df)
    dfs = pl.concat(dfs, how='horizontal')
    score_min = pl.Series(
            dfs.select(cs.matches("score")).to_numpy().min(1))
    dfs = dfs.with_columns(score_min.alias("score_min"))
    pval_max = pl.Series(
            dfs.select(cs.matches("pval")).to_numpy().max(1))
    dfs = dfs.with_columns(pval_max.alias("pval_max"))
    if len(tvals) > 2:
        dfs = regressGammaFromKappa(dfs, np.array(tvals).reshape(-1,1))
    return dfs

def knnImpute(X, k : int=15, metric : str="l2", do_pca : bool = True, n_pcs : int = 50,):
    """
    Impute values based on KNN.
    """
    if do_pca:
        pca = PCA(n_components=n_pcs)
        pca.fit(X)
        Y = pca.transform(X)
    else:
        Y = X
    tree = BallTree(Y, metric=metric)
    kns = tree.query(Y, return_distance=False, k=k)
    Z = X[kns].mean(-2)
    return Z

def vImpute(X, NNS, C=None):
    """
    X : n×g matrix of n observation over g dimensions to be imputed
    NNS: n×k nearest neighbors index matrix (like the output from query tree)
    C: n×n optional weight/connectivities matrix for weighted average. If None will use non-weighted avg.
    """
    if C is None:
        Y = X[NNS].mean(-2)
    else:
        rind, cind = np.indices((C.shape[0], NNS.shape[1]))
        Y = C[rind, NNS, np.newaxis] * X[NNS]
        Y = Y.mean(-2)
    return Y

def simpleNNImpute(
        X : np.ndarray,
        Xp : Optional[np.ndarray] = None,
        n_neighbors : int = 15,
        n_components : int = 30,
        **kwargs,
        ):
    """
    Impute observations (rows) using unweighyed neighbor mean baed on specified parameters

    parameters
    ----------
    `X` : #(n_obs, n_dims) array to be imputed.
    `Xp` : a pca reduction of X can optionaly be precomputed and given to the function otherwise will be created
    `n_neighbors` : number of nearest neighbors
    `n_components` : number of components for pca (if Xp=None)
    `**kwards` : arguments for BallTree if non-default (l2 norm) is desired.

    output
    ------
    M : #(n_obs, d_dims) imputation of X.
    """
    if Xp is None:
        pca = PCA(n_components=n_components).fit(X)
        Xp = pca.fit_transform(X)
    tree = BallTree(X, **kwargs)
    dist, ind = tree.query(k=n_neighbors)
    M = X[ind].mean(-2)
    return M

def logBinCountArray(X, n_bins=11):
    """
    returns for every element in an array its `logbin index` calculated as follows:
    for row i let m_i be the max value. if x[i,j] = 0 its logbin index is 0. Otherwise its index is
    `1 + math.log(x[i,j]) // (math.log(m_i) / (n_bins - 1) )`
    In other words logarithmize the non-0 values in each row and for each value find to which of the equally spaced 
    (n_bins -1 ) bins between 0 to log(m_i) it belongs.

    parameters
    ----------
    `X` : #(n_obs, n_dims) array
    `n_bins` : number of beans to use (11)
    """
    Y = np.log(X * (X > 0))
    m = Y.max(0, keepdims=True) / (n_bins)
    Z = np.floor_divide(Y, m) + 1
    Z = np.nan_to_num(Z, nan=0, neginf=0)
    return Z

def firstMoment(
        adata : ad.AnnData,
        layers : typing.List[str] =[],
        suffix : str="_m",
        method : str="unweighted",
        returnWeights : bool=False,
        ):
    """
    calculate first moment like scVelo or with some
    adaptations. expects adata object to contain .obsp['connectivities']
    (which is the result of running sc.neighbors)
    method : either scvelo style (unweighted avg) or 'weighted' by the connectivities.
    """
    I = np.identity(adata.n_obs)
    C = np.zeros_like(I)
    C[:,:] = unsparse(adata.obsp['connectivities'])
    if method == "scvelo" or method == "unweighted":
        C = (C > 0).astype(float) + I
        #C = (C > 0).astype(float)
        #C = C + I
    else:
        C = I + C
    C = normalize(C, norm='l1', axis=1)
    for layer in layers:
        adata.layers[layer + suffix] = \
                C @ adata.layers[layer]
    if returnWeights:
        return C


def labelCluster(adata, cluster_key : str = 'leiden', label_key : str = 'cell_type', cluster_label : str = 'cluster_label'):
    """
    asign to each cell in a cluster the cell type label of the majority of cell types in that cluster
    """
    d = {}
    #adata.obs[cluster_label] = "unknown"
    for cluster in np.unique(adata.obs[cluster_key]):
        mark_cluster = adata.obs[cluster_key] == cluster
        counter = Counter(adata.obs[mark_cluster][label_key])
        label = counter.most_common(1)[0][0]
        adata.obs[mark_cluster][cluster_label] = label
        d[cluster] = label
        #print(label + " : " + cluster)
    adata.obs[cluster_label] = adata.obs[cluster_key].map(d)
    return d

class Binarizer():
    """
    This class is used to produce binarized data out of real valued non-negative input data which specifically
    RNA expression data in mind.
    By initiating the the user can set the definition of outlier (for larrge values) in terms of multiples of standard deviation
    (defaults to 1.5).
    Values that fall above set value will be clipped to max value.
    The clipped set is then normalized to the range [0,1]. In terms of these normalized values the threshold is 0.5 (or user defined?).
    All observations that are after said process greater than 0.5 will be set to 1 ('ON') and the others to 0 ('OFF').

    The underlying assumption in this procedure is that high expression values represent genes in steady state zone and if the clipping is done right then
    most values whould squeez in the high and low values of the sigmoid curve with transitory values in the linear part. However there is no assumption made about 
    proportion of cells in OFF or ON mode, just that sufficiently many are in ON mode and OFF mode and near the respective steady state.

    for the purpose of outlier detection only positive expression is considered because typically there is a huge inflation of 0
    in the data which might skew the detection of real outlier. (perhaps should 0 up to some max proportion of the set or fix to that proportion?)
    """
    def __init__(self, outlier_mul = 0.5, pct_zeros : float = 0.1):
        self.outlier_mul = outlier_mul
        self.pct_zeros = pct_zeros
        return

    def normalize(self, x : np.ndarray ):
        """
        ignore this one.
        """
        n = np.sum(x.flatten() >0)
        n = int(n)
        m = int( (1 + self.pct_zeros) * n )
        z = np.zeros((m,))
        z[:n] = x.flatten()[x.flatten() > 0]
        std = np.std(z, ddof=1)
        zstd = (z - z.mean()) / std
        zmax = z[zstd < self.outlier_mul].max()
        #z = z.clip(max=zmax)
        return zmax #0.5*zmax is the threshold

    def normalizeArray(self, X : np.ndarray, axis : int = -1, maskZeros : bool = True):
        """
        applies the same normalizing procedure but without pct_zeros on the axis of choice.
        """
        mask = np.ones_like(X).astype(bool)
        if maskZeros:
            mask = X > 0
        std = np.std(X, axis=axis, where=mask, keepdims=True) + 1e-10
        mu = np.mean(X, axis=axis, where=mask, keepdims=True)
        zmax = mu + self.outlier_mul * std
        return zmax

#def normalizeBatches(
#        adata : ad.AnnData,
#        layers : typing.List[str],
#        by_key: typing.Optional[str]="time",
#        size_factor : float=1e5,
#        ):
#    """
#
#    """
#    pass

def standardize(
        adata : ad.AnnData,
        layers : typing.List[str],
        by_key: typing.Optional[str]="time",
        suffix: str="_st",
        do_center: bool=True,
        do_scale: bool=True,
        do_size_normalize: bool=False,
        size_factor : float=1e5,
        ):
    """
    Center and scale the listed layers such that for each gene in each layer, in every sublayer 
    which have the same value corresponding to the given key, has 0 mean and 1 variance (if do_scale=True).
    The result is written into new layers with the added suffix.
    parameters:
    adata : the AnnData object 
    layers: list of layers to standardized
    by_key: the key in the obs dataframe used for the subsetting (default='time')
    suffix: the suffix used for naming the output layers.
    do_center: if True the output is centered.
    do_scale: if Ture (default) the output is scaled, otherwise just centered.
    note if last two are both False nothing will happen.
    """
    if not (do_center or do_scale):
        return
    groups = ['group'] if by_key is None else np.unique(adata.obs[by_key])
    for layer in layers:
        newlayer = layer + suffix
        adata.layers[newlayer] = np.zeros_like(adata.layers[layer]) + adata.layers[layer].copy()
        for g in groups:
            mark = (adata.obs[by_key] == g) if by_key is not None else [True] * adata.n_obs
            if do_size_normalize:
                adata.layers[newlayer][mark] = size_factor * np.nan_to_num(adata.layers[newlayer][mark] / np.nansum(adata.layers[newlayer][mark], axis=1, keepdims=True))
            if do_center:
                adata.layers[newlayer][mark] = adata.layers[newlayer][mark] - np.nanmean(adata.layers[newlayer][mark], axis=0, keepdims=True)
            if do_scale:
                adata.layers[newlayer][mark] = np.nan_to_num(adata.layers[newlayer][mark] / np.nanstd(adata.layers[newlayer][mark], axis=0, keepdims=True))
    return

def find_mnn(
        X=None,
        Y=None,
        metric='minkowski',
        p=2,
        mk=2,
        k=6,
        allow_self : bool=False,
        do_normalize : bool=True,
        **kwargs):
    """
    The assumption is that X, Y represent two modalities of the same set of samples.
    default action:
    Finds for each sample x in X its mutual nearest neighbors in Y
    among its k nearest neighbors in Y.
    parmeters.
    ---------
    X : (m_samples, n_features)
    Y : (m_samples, n_features)
    output
    -------
    mnn : [[int]] : mnn[i] is a list of mnn of i
    mnnn : [[int]] : mnnn[i] is a list of mnn of i
            or [i] if mnn[i] is [].
    """
    if do_normalize:
        X = normalize(X, norm='l2', axis=0)
        Y = normalize(X, norm='l2', axis=0)
    Xtree = BallTree(X, metric=metric, p=2, )
    Ytree = BallTree(Y, metric=metric, p=2, )
    xnn = Ytree.query(X, k=k, return_distance=False,)
    ynn = Xtree.query(Y, k=k, return_distance=False,)
    n = X.shape[0]
    mnn = n * [[]]
    for i in range(n):
        #mnn[i] = [j if i in xnn[j] else -i for j in ynn[i]]
        mnn[i] = [j for j in xnn[i] if (i in ynn[j]) and (allow_self or (i != j))]
    mnnn = [mnn[i] if mnn[i] != [] else [i] for i in range(n)]
    #Xmnn_avg = np.array([X[mnn[i]].mean(0) for i in range(n)])
    #mnn = np.array(mnn)
    return mnn, mnnn

def findMNN(ind0, ind1):
    """
    ind0 : first neighbor index array
    ind1 : second neighbor index array
    returns index array `ind` such that 
    j \in ind[i] iff j \in ind0[i] AND i \in ind1[j]
    """
    n = len(ind0)
    ind = np.nan * np.ones_like(ind0)
    ind = ind0.copy()
    test = np.array(
        [i != j and i in ind1[j] for i in range(n) for j in ind0[i]]
    ).reshape(n,-1)
    ind = np.where(test, ind, np.nan)
    return ind

def getNNfromPWDistance(
        D : np.ndarray,
        n_neighbors : int=15,
        ):
    """
    get sorted k-nearest neighbors and distances from a pairwise distance matrix
    params
    ------
    D : (x_obs , y_obs) 2-d array of pairwise distance D[x,y]=d(x,y) expected to be non-negative (and symmetric) with 0 diagonal.
    n_neighbors : number of nearest neighbors.
    output
    ------
    dd : (n_obs, n_neighbors) array of distances to the nearest neighbors sorted up
    jj : (n_obs, n_neighbors) array of indices of the nearest neighbors jj[x,y] is the index
    of x's y's smallest nearest neighbor.
    """
    n = len(D)
    jj = np.argsort(D, axis=1)[:,:n_neighbors]
    ii = np.arange(n).reshape(-1,1)
    dd = D[ii, jj]
    return dd, jj

def gaussKernelFromDistance(
        X : Optional[np.ndarray]=None,
        Y : Optional[np.ndarray]=None,
        D : Optional[np.ndarray]=None,
        NFilter : Optional[np.ndarray]=None,
        n_neighbors : Optional[int] = 30,
        metric : str = 'euclidean',
        eps = 1e-14,
        scale : float = 4,
        #density_adj : bool = False,
        #symmetric_normalize : bool = False,
        #return_dist : bool = True,
        only_nn : bool=True,
        **kwargs,
        ):
    """
    ----------
    Parameters
    ----------
    `X` : `Optional[np.ndarray]=None`  
    `Y` : `Optional[np.ndarray]=None`  
    `D` : `Optional[np.ndarray]=None`  
    `NFilter` : `Optional[np.ndarray]=None`  
    `n_neighbors` : `Optional[int]=30` 
    `metric` : `str='euclidean'`
    `eps` : `float=30`,
    `scale` : `float=4`
    `only_nn` : `bool=True`

    -------
    output
    -------
    `W` :`#(nx,ny) array` unnormalized Gaussian kernel matrix
    """
    if Y is None:
        Y = X
    if D is None:
        D = pairwise_distances(X, Y=Y, metric=metric, n_jobs=-1, **kwargs)
    nx = D.shape[0]
    ny = D.shape[1]
    jjx = np.argsort(D, axis=1)[:,1:n_neighbors] #(nx, n_neighbors - 1)
    iix = np.arange(nx).reshape(-1,1) #(nx, 1)
    ddx = D[iix,jjx] #(nx, n_neighbors -1)
    sigmax = ddx[:,[-1]] #(nx, 1)
    sigmax_sq = sigmax**2
    iiy = np.argsort(D, axis=0)[1:n_neighbors, :] #(n_neighbors-1, ny)
    jjy = np.arange(ny).reshape(1,-1) #(1, ny)
    ddy = D[iiy,jjy] #(n_neighbors -1, ny)
    sigmay = ddy[[-1], :] #(1, ny)
    sigmay_sq = sigmay**2
    if only_nn:
        W = np.inf * np.ones_like(D)
        W[iix, jjx] = ddx**2
    else:
        W = D ** 2
    if NFilter is not None:
        W = np.where(NFilter, W, np.inf)
    # exclide the major diagonal
    #W[iix, jjy] = np.inf
    W = W / 2 / (sigmax_sq + sigmay_sq)
    W = np.exp(-W * scale**2) * np.sqrt(2 * sigmax_sq * sigmay_sq / (sigmax_sq + sigmay_sq) )
    return W




#import typing
#import numpy as np
#from scipy.spatial.distance import pdist, squareform, cdist
def gaussianKernelImpute(
        XA : np.ndarray,
        XB : typing.Optional[np.ndarray]=None,
        YA : typing.Optional[np.ndarray]=None,
        D_BA : typing.Optional[np.ndarray]=None,
        sigmaA = None,
        sigmaB = None,
        n_neighbors : int=15,
        metric : str='euclidean',
        scale : float = 4,
        kernel : str='gauss',
        remove_first_neighbor : bool = False,
        **kwargs,
        ):
    """
    The basic idea: XA is a marix of observations (na_obs, x_dim), YA is (nb_obs, y_dim) embedding of 
    XA in lower dimension. XB is a matrix of (nb_obs, x_dim) new observations. The goal is to create matrix 
    YB (nb_obs, y_dim) representing the embedding of XB using nearest of XB in XA for weights and the embedding YA.
    Gaussian kernel is used to produce the weights.

    ----------
    parameters
    ----------
    `XA` : input array of na_obs observation of x_dim dimensions
    `XB` : optional input array of nb_obs observation of x_dim dimensions
    `YA` : optional array of (na_obs, y_dim) which is the embedding of XA
    `D_BA` : (nb_obs, na_obs) optionally precomputed pairwise distances
    `sigmaA` : optional array (na_obs) of the std for XA 
    `sigmaB` : optional array (nb_obs) of the std for XB 
    `n_neighbors` : number of nearest neighbors to consider.
    `sclae` : float postive. higher value makes the kernel tighter.
    `metric` : scipy recongnizable metric (defaults to l2)
    `remove_first_neighbor` : bool (False) whether to remove the first nearest neighbor (in cases.
    it is the observation itself)
    `kwargs` : dictionary with additional parameters for the metric (for minkowski metrics).
    ------
    output
    ------
    W_BA : (nb_obs, na_obs) The weights (row normalized)
    YB : (nb_obs, y_dim) The imputed embedding for XB
    """
    if XA.ndim != 2:
        print("X must be 2d array")
        return None
    if XB is None:
        XB = XA
    if YA is None:
        YA = XA
    na_obs = XA.shape[0]
    nb_obs = XB.shape[0]
    x_dim = XA.shape[1]
    y_dim = YA.shape[1]
    if D_BA is None:
        D_BA = pairwise_distances(XB, Y=XA, metric=metric, n_jobs=-1, **kwargs)
    n = D_BA.shape[0]
    m = D_BA.shape[1]
    jjB = np.argsort(D_BA, axis=1)[:,:n_neighbors] # NN of XB in XA
    iiB = np.arange(nb_obs).reshape(-1,1)
    ddB = D_BA[iiB, jjB]
    if remove_first_neighbor:
        ddB = ddB[:,1:]
        jjB = jjB[:,1:]
    if sigmaB is None:
        sigmaB = ddB[:,-1] #(nb_obs,)
    sigmaB = sigmaB.reshape(-1, 1) #(nb_obs, 1)
    sigmaBsq = sigmaB ** 2
    jjA = np.argsort(D_BA.T, axis=1)[:,:n_neighbors] #NN of XA in XB
    iiA = np.arange(na_obs).reshape(-1,1)
    ddA = D_BA.T[iiA, jjA]
    if remove_first_neighbor:
        ddA = ddA[:,1:]
        jjA = jjA[:,1:]
    if sigmaA is None:
        sigmaA = ddA[:,-1] #(na_obs,)
    sigmaA = sigmaA.reshape(1, -1) #(1, na_obs)
    sigmaAsq = sigmaA ** 2
    W_BA = np.inf * np.ones((nb_obs, na_obs))
    W_BA[iiB, jjB] = ddB ** 2
    W_BA = W_BA / 2 / (sigmaAsq + sigmaBsq)
    W_BA = np.exp(-W_BA * scale ** 2)
    W_BA = W_BA * np.sqrt(2 * sigmaA * sigmaB / (sigmaAsq + sigmaBsq) )
    W_BA = W_BA / W_BA.sum(axis=1, keepdims=True) #(nb_obs, na_obs)
    YB = W_BA @ YA #(nb_obs, y_dim)
    return W_BA, YB

def getSymTransitionFromConnectivities(
        adata,
        n_comps : int=15,
        ):
    """
    eponimous. assumes connectivities had been computed and stored in 
    adata.
    params:
    `adata`
    `n_comps : int = 15` : number of components
    output:
    K : symmetric transition from connectivities
    l : n_compps largest eigenvals descending
    Q : n_compps matching eigenvectors (n_obs,n_comps)

    """
    #n = Q.shape[0]
    n = adata.n_obs
    C = adata.obsp['connectivities'].todense().A
    q = C.sum(axis=1, keepdims=True)
    K = C / q / q.T #density normalization
    z = np.sqrt(K.sum(1, keepdims=True))
    K = K / z / z.T #symmetric transition matrix
    l, Q = linalg.eigh(K, subset_by_index=[n-15,n-1], ) #ascending order
    l = l[::-1] #now decending
    Q = Q[:, ::-1] #now decending
    return K, l, Q

def getDptMatrix(l, Q):
    """
    returns the diffution pseudotime matrix `dpt` based 
    on the eigenvals l and eigenvectors Q
    """
    n = Q.shape[0] #n_obs
    k = Q.shape[1] - 1 #n_comps
    I = np.identity(k)
    ll = np.diag(l[1:])
    ll = np.diag( l[1:] / (1 - l[1:]) ) #(k,k)
    psi = Q[:,1:] #(n,k)
    dpt = psi @ ll #(n,k)
    return dpt

def dpt_xy(x : int ,y : int, dpt : np.ndarray):
    """
    dpt distance of `x` from `y` given `dpt` matrix
    """
    #n = Q.shape[0] #n_obs
    #k = Q.shape[1] - 1 #n_comps
    #dpt = getDptMatrix(l, Q) #(k,n)
    dpt_xy = linalg.norm(dpt[x] - dpt[y], ord=2)
    return dpt_xy

def getGaussConnectivities(
        X : typing.Optional[np.ndarray],
        D : typing.Optional[np.ndarray] = None,
        dd : typing.Optional[np.ndarray] = None,
        jj : typing.Optional[np.ndarray] = None,
        n_neighbors : int = 15,
        metric : str = 'euclidean',
        eps = 1e-14,
        scale : float = 4,
        symmetric : bool = True,
        density_adj : bool = False,
        symmetric_normalize : bool = False,
        return_dist : bool = True,
        **kwargs,
        ):
    """
    Calculate the gaussian kernel connectivities for given set of observations.
    Must provide either X (observation matrix), or D (pairwise distances matrix), or (dd, jj)
    nearest neighbor arrays.
    -----------
    parameters
    -----------
    X: optional `array` (n_obs, n_dim) if not provided must provide dd and jj.
    D: optional `array` (n_obs, n_obs) of all pairwise distances (alternative to X) 
    dd : optional (n_obs, n_neighbors) array of nearest neighbor distances
    jj : optional (n_obs, n_neighbors) sorted array of nearest neighbor indices including self!
    n_neighbors: `int` numbers of nearest neighbors for distance matrix. However the final
    output when symmetric=True contains more than number.
    metric: `str` any valid scipy metric
    eps : `float` everything below eps is 0
    scale : `float` (4) the standard deviasion larger means linearly smaller std (narrower bell curve).
    `symmetric` : bool (True) whether to retrun a symmetric kernel
    `density_adj` : bool (False) whether to include density adjacement in the computation. Only makes sense if symmetric=True (I think)
    `symmetric_normalize` : bool (False) whether to normalize the kernel the as a symmetric Laplacian (only makes sense if symmetric=True)
    return_dist : `bool` (True) whether to return dd, jj in addition to K
    **kwarg : additional parameters to pass for the metric
    ------
    output
    ------
    K : (n_obs, n_obs) connectivity matrix 
    dd : diatances (only if return_dist = True) 
    jj : indices (only if return_dist = True)
    """
    if X is not None:
        n = len(X)
        tree = BallTree(X, metric=metric,  **kwargs)
        dd, jj = tree.query(X, k=n_neighbors) #distances, cols
        jj = jj[:, 1:]
        dd = dd[:, 1:]
    elif D is not None:
        n = len(D)
        dd, jj = getNNfromPWDistance(D, n_neighbors)
        dd = dd[:,1:]
        jj = jj[:,1:]
    else: #dd, jj must be provided
        n = len(dd)
    ii = np.arange(jj.shape[0]).reshape(-1,1) #rows
    #sigma = np.median(dd, axis=1).reshape(-1,1) #(obs, 1)
    sigma = dd[:,-1].reshape(-1,1) #(obs, 1)
    sigma_sq = (sigma**2 + sigma.T**2)
    K = np.ones((n,n)) * np.inf
    K[ii, jj] = dd ** 2 / (2 * sigma_sq[ii, jj])
    K = np.exp(-K * scale ** 2)
    K = K * np.sqrt( (2 * sigma * sigma.T) / sigma_sq )
    K = np.where(K < eps, 0, K)
    if symmetric:
        K = np.where(K == 0, K.T, K)
    if density_adj:
        kk = 1 / K.sum(axis=0, keepdims=True)
        K = K * (kk * kk.T)
    if symmetric_normalize:
        z = np.sqrt(K.sum(0, keepdims=True))
        K = K / z / z.T
    return (K, jj, dd) if return_dist else K

def canonicalDPT(
        X : typing.Optional[np.ndarray], # (obs,dim)
        D : typing.Optional[np.ndarray] = None, # (obs,dim)
        dd : typing.Optional[np.ndarray] = None,
        jj : typing.Optional[np.ndarray] = None,
        n_neighbors : int = 15,
        n_comps : int = 20,
        metric : str = 'euclidean',
        scale : float = 4,
        root_index : int = 0,
        eps : float = 1e-14,
        density_adj : bool = False,
        **kwargs,
        ):
    """
    cannonical Diffusion pseudotime. Provide one of X, or D, or (dd and jj)


    params
    ------
    X: optional `array` (n_obs, n_dim) if not provided must provide dd and jj.
    D: optional `array` (n_obs, n_obs) of all pairwise distances (alternative to X) 
    dd : optional (n_obs, n_neighbors) array of nearest neighbor distances
    jj : optional (n_obs, n_neighbors) sorted array of nearest neighbor indices including self!
    n_neighbors: `int` numbers of nearest neighbors for distance matrix. However the final
    output when symmetric=True contains more than number.
    metric: `str` any valid scipy metric
    eps : `float` everything below eps is 0
    scale : `float` (4) the standard deviasion larger means linearly smaller std (narrower bell curve).
    `roo_index` : index of the dpt source (time 0)
    `density_adj` : bool (True) whether to include density adjacement in the computation
    **kwarg : additional parameters to pass for the metric

    output
    ------
    `M` : (n_obs, n_obs) the diffusion matrix
    `dpt` : (n_obs,) pseudotime from root index to all obs.
    """
    K, jj, dd = getGaussConnectivities(
            X, D, dd, jj, scale=scale,
            n_neighbors=n_neighbors, metric=metric, symmetric=True,
            eps=eps, return_dist=True,
            **kwargs,
            )
    n = len(K)
    if density_adj:
        kk = 1 / K.sum(axis=0, keepdims=True)
        K = K * (kk * kk.T)
    # normalize (symmetric Laplacian style):
    Z = np.sqrt(K.sum(0, keepdims=True))
    T = K / Z / Z.T
    # eigen decomposition
    l, Q = linalg.eig(T,)
    # drop 1st eign vect, take the next n_comps
    psi = Q[:, 1 : n_comps]
    lmb = l[1 : n_comps]
    lmb = lmb / (1 - lmb)
    # The obtained diffusion matrix
    M = lmb.reshape(1,-1) * psi @ psi.T
    # diffusion distance from root:
    ps = np.zeros_like(M)
    ps[root_index, :] = -1
    ps = np.identity(n) + ps
    dpt = np.sqrt(((M.T @ M @ ps) * ps ).sum(0))
    dpt[root_index] = dpt.mean()
    dpt[root_index] = dpt.min()
    return M, dpt

def canonicalDPTImproved(
        X : typing.Optional[np.ndarray], # (obs,dim)
        D : typing.Optional[np.ndarray] = None, # (obs,dim)
        dd : typing.Optional[np.ndarray] = None,
        jj : typing.Optional[np.ndarray] = None,
        n_neighbors : int = 15,
        n_comps : int = 20,
        metric : str = 'euclidean',
        scale : float = 4,
        root_index : int = 0,
        eps : float = 1e-14,
        density_adj : bool = False,
        symmetric : bool = False,
        **kwargs,
        ):
    """
    cannonical Diffusion pseudotime. Provide one of X, or D, or (dd and jj)


    params
    ------
    X: optional `array` (n_obs, n_dim) if not provided must provide dd and jj.
    D: optional `array` (n_obs, n_obs) of all pairwise distances (alternative to X) 
    dd : optional (n_obs, n_neighbors) array of nearest neighbor distances
    jj : optional (n_obs, n_neighbors) sorted array of nearest neighbor indices including self!
    n_neighbors: `int` numbers of nearest neighbors for distance matrix. However the final
    output when symmetric=True contains more than number.
    metric: `str` any valid scipy metric
    eps : `float` everything below eps is 0
    scale : `float` (4) the standard deviasion larger means linearly smaller std (narrower bell curve).
    `roo_index` : index of the dpt source (time 0)
    `density_adj` : bool (True) whether to include density adjacement in the computation
    **kwarg : additional parameters to pass for the metric

    output
    ------
    `M` : (n_obs, n_obs) the diffusion matrix
    `dpt` : (n_obs,) pseudotime from root index to all obs.
    """
    K, jj, dd = getGaussConnectivities(
            X, D, dd, jj, scale=scale,
            n_neighbors=n_neighbors, metric=metric, symmetric=True,
            eps=eps, return_dist=True,
            **kwargs,
            )
    n = len(K)
    if density_adj:
        kk = 1 / K.sum(axis=0, keepdims=True)
        K = K * (kk * kk.T)
    # normalize (symmetric Laplacian style):
    Z = np.sqrt(K.sum(0, keepdims=True))
    T = K / Z / Z.T # the symmetric 'transition' matrix
    P = K / K.sum(1, keepdims=True) # the row normalized transition matrix
    # eigen decomposition
    l, Q = linalg.eigh(T,)
    sorted_indices = np.argsort(l)[::-1] #descending order
    l = l[sorted_indices]
    Q = Q[:,sorted_indices]
    # drop 1st eign vect, take the next n_comps
    psi = Q[:, 1 : n_comps]
    if symmetric == False:
        psi = psi / Z.T
    lmb = l[1 : n_comps]
    lmb = lmb / (1 - lmb)
    # The obtained diffusion matrix
    M = lmb.reshape(1,-1) * psi
    # diffusion distance from root:
    dpt = np.linalg.norm(M - M[[root_index],:], axis=1)
    return M, dpt


# currently doesn't do more than cannonicalDPT
# remove it ?
def generalizedDPT(
        X : np.ndarray,
        n_neighbors : int = 15,
        n_comps : int = 20,
        metric : str = 'euclidean',
        root_index : int = 0,
        eps = 1e-14,
        **kwargs,
        ):
    """
    generalized DPT on observations X (n_obs, n_dim)
    """
    n = len(X)
    tree = BallTree(X, metric=metric, **kwargs)
    dd, jj = tree.query(X, k=n_neighbors) #distances, cols
    ii = np.arange(jj.shape[0]).reshape(-1,1) #rows
    sigma = np.median(dd, axis=1).reshape(-1,1) #(obs, 1)
    sigma_sq = (sigma**2 + sigma.T**2)
    K = pairwise_distances(X, metric=metric, n_jobs=-1, **kwargs)
    K = np.exp(- K**2 / (2 * sigma_sq) )
    K = K * np.sqrt( (2 * sigma * sigma.T) / sigma_sq )
    K[ii,ii] = 0
    K = np.where(K < eps, 0, K)
    # Density normalize:
    D = np.diag(1 / np.nansum(K, axis=0))
    W = D @ K @ D
    # symmetric normalization
    #Z = np.sqrt(W.sum(0, keepdims=True))
    #W = W / Z / Z.T
    Z = np.nan_to_num(np.diag(1 / np.sqrt(W.sum(0, ) ) ))
    W = Z @ W @ Z
    # eigen decomposition
    l, Q = linalg.eig(W,)
    # drop 1st eign vect, take the next n_comps
    psi = Q[:, 1 : n_comps]
    lmb = l[1 : n_comps]
    lmb = lmb / (1 - lmb)
    # The obtained diffusion matrix
    M = lmb.reshape(1,-1) * psi @ psi.T
    # diffusion distance from root:
    ps = np.zeros_like(M)
    ps[root_index, :] = -1
    ps = np.identity(n) + ps
    dpt = np.sqrt(((M.T @ M @ ps) * ps ).sum(0))
    dpt[root_index] = dpt.mean()
    dpt[root_index] = dpt.min()
    return K, M, dpt


def generalizedDPTMatrix(A : np.ndarray, isColTransition : bool = False):
    """
    parameters
    ----------
    `A` : `n × n` non-negative weighted adjacency matrix. A doesn't have to be symmetric.
    However it is assumed that the graph is strongly connected and aperiodic which is equivalent to A being
    irreducible and primitive.
    output
    `isColTransition` : if True, indicates that A is not adjacency but already l1 normalized on the columns 
    ------
    `M` : n × n matrix defined as follows:
    Let T be the column normalized transition matrix induced by A. Let G be the projection on T's 
    largest right eigen vector (corresponding to 1).
    Let K = T - G,
    then let M = \sum_{k=0}^{\inf} K^k = (I - K)^{-1}
    M is used to calculate the dpt distance defined as
    dpt(x,y) = \|M(x-y)\|
    """
    if isColTransition:
        T = A
    else:
        T = A / A.sum(axis=0, keepdims=True) #T.sum(0)=1 transition on column vectors.
    l, Q = linalg.eig(T, )
    P = linalg.inv(Q)
    # P = Q^{-1}
    # P @ T @ Q = diag(l)
    G = Q[: , [0]] @ P[[0], :]
    K = T - G
    I = np.identity(len(T))
    M = linalg.inv(I - K) - I
    return M, K, l

# Brigitte's label transfer for HSCs datasets
def label_transfer(adata, batch_key="time", basis='X_scanorama_reduced', label_key="clusters",
                   reference="control", query="3h", no_neighbours=10):
    '''
    Function to transfer labels from a reference to a query. 
    Query and reference should both be included in one
    Anndata object. 
    
    Parameters
    ----------
    adata
        Annotated data matrix
    batch_key: `str` (default: "time")
        The name of the column in adata.obs that differentiates 
        reference from query
    basis: `str` (default: "X_scanorama_reduced")
        The name of the matrix in adata.obsm that is used to 
        calculate distance between cells
    label_key: `str` (default: "clusters")
        The name of the column in adata.obs which contains 
        the labels that have to be transferred
    reference: `str` (default: "control")
        The name that seperates the reference from the query
        in the adata.obs column indicated using batch_key
    query
        The name that seperates the query from the 
        reference in the adata.obs column indicated using 
        batch_key
    no_neighbours: `int` (default: 10)
        Number of neighbours to use for data integration
    '''

    distances = scipy.spatial.distance.cdist(adata[adata.obs[batch_key]==reference].obsm[basis],
                                             adata[adata.obs[batch_key]==query].obsm[basis],
                                             metric='euclidean')
    df_distances = pd.DataFrame(distances,
                                index=adata[adata.obs[batch_key]==reference].obs[label_key],
                                columns=adata[adata.obs[batch_key]==query].obs_names)
    neighbours = df_distances.apply(lambda x: pd.Series(x.nsmallest(no_neighbours).index))
    transferred_labels = neighbours.value_counts().idxmax()
    transferred_labels = pd.Series(transferred_labels, dtype="category")
    transferred_labels.index = adata[adata.obs[batch_key]==query].obs_names

    return transferred_labels

class OrderedDefaultDict(OrderedDict):
    def __init__(self, default_factory=None, *args, **kwargs):
        if default_factory is not None and not callable(default_factory):
            raise TypeError("default_factory must be callable")
        self.default_factory = default_factory
        super().__init__(*args, **kwargs)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        # Assign the default value and return it
        self[key] = value = self.default_factory()
        return value

def cluster_annotate(
        adata : ad.AnnData,
        cluster_key : Optional[str],
        cell_type_scores : typing.List[str],
        new_col_name : Optional[str] = None,
        #new_col_name : Optional[str] = 'cell_type',
        ):
    """
    Assign cell type to each cluster based on maximally expressed cell type score in that cluster.

    Parameters
    ----------
    adata  
        AnnData type whose obs contain a numerical columns with matching names
        to each of the cell type scores in cell type list. It must also contain column matching the cluster_key.
    cluster_key
        name of the cluster column in adata.obs
    cell_type_scores
        list of columns in adata.obs which contain the cell type scores for each cell type for each cell.
    new_col_name
        if not None updates adata with the assigned cell types in a new column of that name.

    returns
    -------
    a dictionary which maps for each cluster its matching cell type.
    If new_col_name is not None also adds a column with assigned cell types
    """
    score_array = np.array(adata.obs[cell_type_scores])
    cts_idx = dict(zip(
        range(len(cell_type_scores)),
        cell_type_scores,
        ))
    cluster_ct_mapping = OrderedDefaultDict(lambda : "null")
    if cluster_key is not None:
        for cluster in list(np.unique(adata.obs[cluster_key])):
            mark = adata.obs[cluster_key] == cluster
            ct_i = score_array[mark].sum(0).argmax()
            cluster_ct_mapping[cluster] = cts_idx[ct_i]
    else:
        adata.obs['sc_ct'] = [cts_idx[ct_i] for ct_i in score_array.argmax(1)]
    return cluster_ct_mapping if cluster_key else None


def pcaScore(X, V, eps=1e-12, n_components=30,):
    """
    """
    pca = PCA(n_components=n_components).fit(X)
    Xp = pca.transform(X)
    Y = X + X
    Yp = pca.transform(Y)
    score = np.linalg.norm(Yp, ord=2, axis=-1)
    score = score / (eps + np.linalg.norm(Y, ord=2, axis=1))
    return score

def tf_idf(
    bow : np.ndarray,
    log_normalize : bool=True,
    pc : float = 1,
):
    """
    perform tf-idf transformmation on input BoW matrix
    parameters:
    `bow` : (n_docs, n_vocab) bag of words (counts/unnormalized frequencies) representation 
    of the data.
    `log_normalize` : if true will log(1+x) transform the frequencies
    """
    n_docs, n_vocab = bow.shape
    doc_freq = (bow>0).sum(0, keepdims=True) #(1, n_vocab)
    idf = np.log(n_docs + 1) - np.log(doc_freq + 1)
    tf = np.log(pc + bow) if log_normalize else bow
    tf_idf = tf * idf
    return tf_idf

#####################

def projectVelocity(X_0, X_1, Y_0, k=8, metric='minkowski', p=2, smooth : bool=False,):
    """
    Project velocity (more accurately displacement) for each observation into an 
    an embedding Y based on high dimensional inflation X_0, X_1.
    -----
    Input
    -----
    `X_0` : `(n_obs, n_dims)` representation of the "past" state in high dimension.
    `X_1` : `(n_obs, n_dims)` representation of the "future" states.
    `Y_0` : `(n_ob, n_embd)` a low dimensional embedding of `X_0`.
    `k` : `int` number of nearest neighbors.
    `smoot` : `bool (False)` if True smooth the projected velocities by taking mean over k nearest neighbors.
    ------
    output
    ------
    `V` : `(n_obs, n_embd)` Displacement vecotr such that (Y_0 + V) is an embedding of X_1 based on the neigbor algorithms descrebed next.
    ---------
    algorithm
    ---------
    The embedding of observation i, Y_1[i] is
    V[i] = mean{Y_0[j] : j in nn_k(X_1[i], X_0)} - mean{Y_0[j] : j in nn_k(X_0[i], X_0)}
    Y_1 = Y_0 + V
    where nn_k(X_l[i], X_0) is the set of k nearest neighbors of X_l[i] (l=1, 0) in X_0.
    """
    tree = BallTree(X_0, metric=metric, p=p,) 
    ind = tree.query(X_1, k=k, return_distance=False)
    ind0 = tree.query(X_0, k=k, return_distance=False)
    V = Y_0[ind[:,1:]].mean(1) - Y_0[ind0[:,1:]].mean(1)
    if smooth:
        V = V[ind0].mean(1)
    return V

def createJointAndata(
    X : np.ndarray,
    Y : np.ndarray,
    n_components : int=50,
    k : int = 16,
    adata =None,
    estimate_drift : bool = True,
):
    """
    create a new anndata for joint emmbedding.
    X and Y are assumed to be preprocessed and pca ready (filtered, centered scaled etc.)
    """
    n = X.shape[0]
    dX = Y - X
    XY = np.concatenate((X,Y), axis=0)
    pca = PCA(n_components=n_components,).fit(XY)
    Xp = pca.transform(X)
    Yp = pca.transform(Y)
    tree = BallTree(Xp, metric='minkowski', p=2)
    dist, ind = tree.query(Xp, k=k)
    Z = X + dX[ind].mean(1) if estimate_drift else Y
    Zp = pca.transform(Z)
    bdata = sc.AnnData(
        X = np.concatenate((X.copy(),Z.copy()), axis=0),
        )
    bdata.uns['ind'] = ind
    return bdata

def createJE(
    adata,
    X_key : str,
    Y_key : str,
    n_components : int =50,
    n_neighbors : int = 20,
    k : int = 16,
    do_umap : bool = True,
    do_drawgraph : bool = True,
    do_diffmap : bool = True,
    smooth : bool = False,
    estimate_drift : bool = True,
):
    """
    Create Joint embedding from two comparable layers.
    Adds PCA velocity and optionally also UMAP, diffmap, draw_graph.
    -----
    Input
    -----
    `X_key` : layer with the 'old' or 'spliced' cell state (after proper transformation)
    `Y_key` : layer with the 'total' cell state (after proper transformation)
    `n_components` (50): PCA components
    `n_neighbors` (20): number of nearest neighbors for the embedding calculations. 
    `k` (16): nearest neighbors for drift/smooth calculation.
    `do_umap` : `bool` (True)
    `do_drawgraph` : `bool` (False)
    `do_diffmap` : `bool` (False) 
    `smooth` : `bool` If True smoothen the velocity in the embedding by taking mean over neighbors
    `estimate_drift` (True): If True estimates drift in the PCA space, otherwise uses the single cell displacement as is.
    """
    X = adata.layers[X_key].copy()
    Y = adata.layers[Y_key].copy()
    n = adata.n_obs
    bdata = createJointAndata(X, Y, n_components, k=k, adata=adata, estimate_drift=estimate_drift)
    sc.pp.scale(bdata, max_value=10,)
    sc.pp.pca(bdata, n_comps=n_components,)
    adata.obsm['X_je_pca'] = bdata[:n].obsm['X_pca'].copy()
    adata.obsm['velocity_je_pca'] = bdata[n:].obsm['X_pca'] - bdata[:n].obsm['X_pca']
    if smooth:
        ind = bdata.uns['ind']
        adata.obsm['velocity_je_pca'] = adata.obsm['velocity_je_pca'][ind].mean(1)
    sc.pp.neighbors(bdata, n_neighbors=n_neighbors,)
    sc.tl.leiden(bdata, flavor="igraph", n_iterations=2, directed=False)
    if do_umap:
        sc.tl.umap(bdata,)
        adata.obsm['X_je_umap'] = bdata[:n].obsm['X_umap'].copy()
        adata.obsm['velocity_je_umap'] = bdata[n:].obsm['X_umap'] - bdata[:n].obsm['X_umap']
        if smooth:
            ind = bdata.uns['ind']
            adata.obsm['velocity_je_umap'] = adata.obsm['velocity_je_umap'][ind].mean(1)
    if do_drawgraph:
        sc.tl.draw_graph(bdata)
        adata.obsm['X_je_draw_graph_fa'] = bdata[:n].obsm['X_draw_graph_fa'].copy()
        adata.obsm['velocity_je_draw_graph_fa'] = bdata[n:].obsm['X_draw_graph_fa'] - bdata[:n].obsm['X_draw_graph_fa']
        if smooth:
            ind = bdata.uns['ind']
            adata.obsm['velocity_je_draw_graph_fa'] = adata.obsm['velocity_je_draw_graph_fa'][ind].mean(1)
    if do_diffmap:
        sc.tl.diffmap(bdata, )
        adata.obsm['X_je_diffmap'] = bdata[:n].obsm['X_diffmap'].copy()
        adata.obsm['velocity_je_diffmap'] = bdata[n:].obsm['X_diffmap'] - bdata[:n].obsm['X_diffmap']
        if smooth:
            ind = bdata.uns['ind']
            adata.obsm['velocity_je_diffmap'] = adata.obsm['velocity_je_diffmap'][ind].mean(1)

##### test
from scipy.stats import rankdata

def quantile_normalize_paired_matrices(X, Y):
    n, p = X.shape
    assert Y.shape == (n, p), "X and Y must have the same shape"
    
    data = np.stack([X, Y], axis=1)  # Shape: (n, 2, p)
    sorted_data = np.sort(data, axis=0)  # Shape: (n, 2, p)
    mean_quantiles = np.mean(sorted_data, axis=1)  # Shape: (n, p)
    
    X_norm = np.zeros_like(X)
    Y_norm = np.zeros_like(Y)
    
    for j in range(p):
        # Compute ranks with ties (average method)
        ranks_x = rankdata(X[:, j], method='average') - 1  # 0-based indexing
        ranks_y = rankdata(Y[:, j], method='average') - 1
        ranks_x_min = rankdata(X[:, j], method='min') - 1  # 0-based indexing
        ranks_x_max = rankdata(X[:, j], method='max') - 1  # 0-based indexing
        ranks_y_min = rankdata(Y[:, j], method='min') - 1
        ranks_y_max = rankdata(Y[:, j], method='max') - 1
        # Assign mean quantiles
        for rank in np.unique(ranks_x):
            mask = ranks_x == rank
            start = ranks_x_min[mask].min()
            stop = ranks_x_max[mask].max()+1
            X_norm[mask,j] = mean_quantiles[start:stop,j].mean()
        for rank in np.unique(ranks_y):
            mask = ranks_y == rank
            start = ranks_y_min[mask].min()
            stop = ranks_y_max[mask].max()+1
            Y_norm[mask,j] = mean_quantiles[start:stop,j].mean()
    return X_norm, Y_norm

def qNormalize(X):
    """
    X (n_obs, n_groups)
    """
    quantiles = np.sort(X,0).mean(1)
    X_norm = np.zeros_like(X)
    for g in range(X.shape[1]):
        ranks_x_min = rankdata(X[:,g], 'min') - 1
        ranks_x_max = rankdata(X[:,g], 'max') - 1
        for j in range(X.shape[0]):
            start = ranks_x_min[j]
            steps = ranks_x_max[j] - ranks_x_min[j] + 1
            stop = start + steps
            X_norm[j,g] = quantiles[start:stop].mean()
    return X_norm
