import polars as pl
import polars.selectors as cs
import anndata as ad
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
import typing
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import seaborn as sns
import slamutils
from slamutils import unsparse


def scatterLegendCbar(
        x,
        y,
        labels : typing.List[str],
        colors : typing.List[str],
        ltitle : typing.Optional[str] = None,
        ax = None,
        fig = None,
        title=None,
        xlabel = None,
        ylabel = None,
        dpi : int=150,
        figsize = (15,10),
        cbar : bool = True,
        **kwargs,
        ):
    """
    Function to scatter plot with legend and colorbar
    https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/scatter_with_legend.html
    https://www.statology.org/matplotlib-scatterplot-legend/
    """
    if not ax:
        fig, ax = plt.subplots(dpi=dpi, figsize=figsize,)
    scatter = ax.scatter(x,y, c=colors, **kwargs)
    legend = ax.legend(handles = scatter.legend_elements()[0],
                       labels = labels,
                       loc="lower left", 
                       title=ltitle,
                       )
    if legend:
        ax.add_artist(legend)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if cbar:
        fig.colorbar(scatter, ax=ax,)
    return fig, ax

def plotPhasePlanesM(
    xdata: ad.AnnData,
    ydata: typing.Optional[ad.AnnData],
    x_layer: str,
    y_layer: str,
    x_vars: typing.List[typing.Union[str, int]],
    y_vars: typing.Optional[typing.List[typing.Union[str, int]]],
    name_col: str,
    color_dict: typing.Optional[dict] = None,
    title: typing.Optional[str] = None,
    ncol: int = 4,
    figsize: typing.Tuple[int] = None,
    s: int = 5,
    xscale: str = "linear",
    yscale: str = "linear",
    aspect: str = "equal",
    otherprops: dict = {},
    showlegend=True,
    cmap : str='viridis',
    dpi : int=250,
    plotMeans : bool=False,
    logTransform : bool=False,
    kappa : typing.Optional[np.ndarray] = None,
    reverse : bool = False,
):
    """
    Produces a grid of phase plane plots.
    xdata and ydata (see below) should be 'in sync'--
    their rows must match and ech row must correspond to the
    same sample.
    Parameters
    ----------
    xdata : AnnData which contains the first (x) modality
    ydata : AnnData which contains the second (y) modality
    x_layer : str indicating the layer of interes in xdata
    y_layer : str indicating the layer of interes in ydata
    x_vars : list indicating the x variables of interest
    y_vars : list indicating the y variables of interest
     x_vars and y_vars should be of equal length and indicate the pairs of interesty
     to be plotted.
    name_col : key to column of interest in xdata.obs (which should appear also in ydata)
    color_dict: a dictionaty with keys in 'name_col' and values indicating color.
    title: plots title
    ncol: number of columns in the frid plot
    figsize: passed to pyplot figsize property
    s: passed as point size (s) pyplot property
    xscale: pyplot 'scale' for x axis
    yscale: pyplot 'scale' for x yxis
    aspect: aspect ratio of the plot.
    otherprops: dictionary with additional pyplot properties.
    """
    if not ydata:
        ydata = xdata
    # n = len(match_list) # number of plots
    n = len(x_vars)
    # nrow = max(2,math.ceil(n / ncol))
    nrow = math.ceil(n / ncol)
    names = list(np.unique(xdata.obs[name_col]))
    if reverse:
        names.reverse()
    if not np.isreal(names[0]):
        names = [str(x) for x in names]
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=figsize,
        dpi=dpi,
    )
    if not color_dict:
        myCmap = sns.color_palette(palette=cmap, n_colors=len(names)).as_hex()
        color_dict = dict(
            zip(
                names,
                myCmap,
            ),
        )
    for k in range(n):
        x = unsparse(xdata[:, x_vars[k]].layers[x_layer]).flatten()
        y = unsparse(ydata[:, y_vars[k]].layers[y_layer]).flatten()
        if logTransform:
            #x = np.log(x + 1e-6)
            #y = np.log(y + 1e-6)
            x = np.log(x + 1)
            y = np.log(y + 1)
        if axes.ndim == 2:
            ax = axes[k // ncol, k % ncol]
        else:
            ax = axes[max(k // ncol, k % ncol)]
        for name in names:
            if plotMeans:
                ax.scatter(
                    x[xdata.obs[name_col] == name].mean(keepdims=True),
                    y[ydata.obs[name_col] == name].mean(keepdims=True),
                    color = color_dict[name],
                    label=name,
                    s=s*5,
                    marker='x',
                    **otherprops,
                )
            else:
                ax.scatter(
                    x[xdata.obs[name_col] == name],
                    y[ydata.obs[name_col] == name],
                    color = color_dict[name],
                    label=name,
                    s=s,
                    **otherprops,
                )
            ax.set_xlabel(x_layer + ":" + x_vars[k])
            ax.set_ylabel(y_layer + ":" + y_vars[k])
            ax.set_aspect(aspect)
            ax.set_xscale(
                xscale,
            )
            ax.set_yscale(
                yscale,
            )
            if k==0 and showlegend:
                ax.legend()
                ax.set_title(title)
            #ax.legend()
            #ax.set_title(title)
    return fig, axes

def plotPhasePlanesAndKappa(
    xdata: ad.AnnData,
    ydata: typing.Optional[ad.AnnData],
    x_layer: str,
    y_layer: str,
    x_vars: typing.List[typing.Union[str, int]],
    y_vars: typing.Optional[typing.List[typing.Union[str, int]]],
    name_col: str,
    color_dict: typing.Optional[dict] = None,
    title: typing.Optional[str] = None,
    ncol: int = 4,
    figsize: typing.Tuple[int] = None,
    s: int = 5,
    xscale: str = "linear",
    yscale: str = "linear",
    aspect: str = "equal",
    otherprops: dict = {},
    showlegend=True,
    cmap : str='viridis',
    dpi : int=250,
    plotMeans : bool=False,
    logTransform : bool=False,
    gkey : typing.Optional[str] = None, 
    chase : bool = False,
    kkey : typing.Optional[str] = None, 
):
    """
    Produces a grid of phase plane plots.
    xdata and ydata (see below) should be 'in sync'--
    their rows must match and ech row must correspond to the
    same sample.
    Parameters
    ----------
    xdata : AnnData which contains the first (x) modality
    ydata : AnnData which contains the second (y) modality
    x_layer : str indicating the layer of interes in xdata
    y_layer : str indicating the layer of interes in ydata
    x_vars : list indicating the x variables of interest
    y_vars : list indicating the y variables of interest
     x_vars and y_vars should be of equal length and indicate the pairs of interesty
     to be plotted.
    name_col : key to column of interest in xdata.obs (which should appear also in ydata)
    color_dict: a dictionaty with keys in 'name_col' and values indicating color.
    title: plots title
    ncol: number of columns in the frid plot
    figsize: passed to pyplot figsize property
    s: passed as point size (s) pyplot property
    xscale: pyplot 'scale' for x axis
    yscale: pyplot 'scale' for x yxis
    aspect: aspect ratio of the plot.
    otherprops: dictionary with additional pyplot properties.
    """
    if not ydata:
        ydata = xdata
    # n = len(match_list) # number of plots
    n = len(x_vars)
    # nrow = max(2,math.ceil(n / ncol))
    nrow = math.ceil(n / ncol)
    names = list(np.unique(xdata.obs[name_col]))
    if not np.isreal(names[0]):
        names = [str(x) for x in names]
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=figsize,
        dpi=dpi,
    )
    if not color_dict:
        myCmap = sns.color_palette(palette=cmap, n_colors=len(names)).as_hex()
        color_dict = dict(
            zip(
                names,
                myCmap,
            ),
        )
    for k in range(n):
        x = unsparse(xdata[:, x_vars[k]].layers[x_layer]).flatten()
        y = unsparse(ydata[:, y_vars[k]].layers[y_layer]).flatten()
        if logTransform:
            x = np.log(x + 1e-6)
            y = np.log(y + 1e-6)
        if axes.ndim == 2:
            ax = axes[k // ncol, k % ncol]
        else:
            ax = axes[max(k // ncol, k % ncol)]
        if kkey:
            kappa = xdata[:, x_vars[k]].var[kkey].to_numpy()[0]
            ax.plot(x, kappa * x, color='black', ls='--',)
        for name in names:
            if plotMeans:
                ax.scatter(
                    x[xdata.obs[name_col] == name].mean(keepdims=True),
                    y[ydata.obs[name_col] == name].mean(keepdims=True),
                    color = color_dict[name],
                    label=name,
                    s=s*5,
                    marker='x',
                    **otherprops,
                )
            else:
                ax.scatter(
                    x[xdata.obs[name_col] == name],
                    y[ydata.obs[name_col] == name],
                    color = color_dict[name],
                    label=name,
                    s=s,
                    **otherprops,
                )
                if gkey:
                    xg = x[xdata.obs[name_col] == name]
                    gamma = xdata[xdata.obs[name_col] == name, x_vars].var[gkey][k]
                    t = float(name)
                    kappa = 1 - np.exp(-gamma * t)
                    if chase: 
                        kappa = 1 - kappa
                    z = kappa * xg
                    ax.plot(xg,z, ls='--', color='black',)
            ax.set_xlabel(x_layer + ":" + x_vars[k])
            ax.set_ylabel(y_layer + ":" + y_vars[k])
            ax.set_aspect(aspect)
            ax.set_xscale(
                xscale,
            )
            ax.set_yscale(
                yscale,
            )
            if k==0 and showlegend:
                ax.legend()
                ax.set_title(title)
            #ax.legend()
            #ax.set_title(title)
    return fig, axes

def plotPhasePlanesOne(
    xdata: ad.AnnData,
    ydata: typing.Optional[ad.AnnData],
    x_layer: str,
    y_layer: str,
    x_var: typing.Union[str, int],
    y_var: typing.Optional[typing.Union[str, int]],
    name_col: str,
    color_dict: typing.Optional[dict] = None,
    title: typing.Optional[str] = None,
    ncol: int = 4,
    figsize: typing.Tuple[int] = None,
    s: int = 5,
    xscale: str = "linear",
    yscale: str = "linear",
    aspect: str = "equal",
    otherprops: dict = {},
    showlegend=True,
    cmap : str='viridis',
    dpi : int=250,
    plotMeans : bool=False,
    logTransform : bool=False,
    kappa : typing.Optional[np.ndarray] = None,
    reverse : bool = False,
    classcol : typing.Optional[str] = None,
):
    """
    Produces a grid of phase plane plots.
    xdata and ydata (see below) should be 'in sync'--
    their rows must match and ech row must correspond to the
    same sample.
    Parameters
    ----------
    xdata : AnnData which contains the first (x) modality
    ydata : AnnData which contains the second (y) modality
    x_layer : str indicating the layer of interes in xdata
    y_layer : str indicating the layer of interes in ydata
    x_var : indicating the x variable of interest
    y_var : indicating the y variable of interest
    name_col : key to column of interest in xdata.obs (which should appear also in ydata)
    color_dict: a dictionaty with keys in 'name_col' and values indicating color.
    title: plots title
    ncol: number of columns in the frid plot
    figsize: passed to pyplot figsize property
    s: passed as point size (s) pyplot property
    xscale: pyplot 'scale' for x axis
    yscale: pyplot 'scale' for x yxis
    aspect: aspect ratio of the plot.
    otherprops: dictionary with additional pyplot properties.
    class :  optional str indicating class/cell_type etc.
    """
    if not ydata:
        ydata = xdata
    names = list(np.unique(xdata.obs[name_col]))
    if classcol:
        class_names = list(np.unique(xdata.obs[classcol]))
        #class_names = [str(s) for s in class_names]
        class_colors = sns.color_palette(palette=cmap, n_colors=len(class_names)).as_hex()
        class_colordict = dict(zip(class_names, class_colors))
    if reverse:
        names.reverse()
    n = len(names) + 1
    nrow = math.ceil(n / ncol)
    if not np.isreal(names[0]):
        names = [str(x) for x in names]
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=figsize,
        dpi=dpi,
    )
    if not color_dict:
        myCmap = sns.color_palette(palette=cmap, n_colors=len(names)).as_hex()
        color_dict = dict(
            zip(
                names,
                myCmap,
            ),
        )
    x = unsparse(xdata[:, x_var].layers[x_layer]).flatten()
    y = unsparse(xdata[:, y_var].layers[y_layer]).flatten()
    k = n-1
    if axes.ndim == 2:
        ax = axes[k // ncol, k % ncol]
    else:
        ax = axes[max(k // ncol, k % ncol)]
    for name in names:
        if plotMeans:
                ax.scatter(
                x[xdata.obs[name_col] == names[k]].mean(keepdims=True),
                y[ydata.obs[name_col] == names[k]].mean(keepdims=True),
                color = color_dict[name],
                label=name,
                s=s*5,
                marker='x',
                **otherprops,
                )
        else:
                ax.scatter(
                x[xdata.obs[name_col] == name],
                y[ydata.obs[name_col] == name],
                color = color_dict[name],
                label=name,
                s=s,
                **otherprops,
                )
    ax.set_xlabel(x_layer + ":" + x_var)
    ax.set_ylabel(y_layer + ":" + y_var)
    ax.set_aspect(aspect)
    ax.set_xscale(
        xscale,
    )
    ax.set_yscale(
        yscale,
    )
    ax.set_title(title)
    ax.legend()
    for k in range(n-1):
        if logTransform:
            x = np.log(x + 1e-6)
            y = np.log(y + 1e-6)
        if axes.ndim == 2:
            ax = axes[k // ncol, k % ncol]
        else:
            ax = axes[max(k // ncol, k % ncol)]
        scatter = ax.scatter(
            x[xdata.obs[name_col] == names[k]],
            y[ydata.obs[name_col] == names[k]],
            #color = color_dict[names[k]],
            #c = xdata[xdata.obs[name_col] == names[k]].obs[classcol].map(class_colordict),
            c = xdata[xdata.obs[name_col] == names[k]].obs[classcol],
            #label=names[k],
            s=s,
            marker='x',
            **otherprops,
            )
        legend = ax.legend(handles = scatter.legend_elements()[0],
                               labels = class_names,
                               loc="lower right",
                               title = classcol,
                               )
        ax.add_artist(legend)
        ax.set_xlabel(x_layer + ":" + x_var)
        ax.set_ylabel(y_layer + ":" + y_var)
        ax.set_aspect(aspect)
        ax.set_xscale(
            xscale,
        )
        ax.set_yscale(
            yscale,
        )
        ax.set_title(title + "_" + str(names[k]))
    return fig, axes

def plotGammaRegression(
        genes,
        slopes,
        intercepts,
        means,
        ts,
        rvals = None,
        ncols : int=3,
        figsize=(15,9),
        dpi : int=250,
        ):
    n = len(genes)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        dpi=dpi,
    )
    for k in range(n):
        if axes.ndim == 2:
            ax = axes[k // ncols, k % ncols]
        else:
            ax = axes[max(k // ncols, k % ncols)]
        g = genes[k]
        ax.scatter(ts, means[k], label=g)
        ax.plot(ts, -slopes[k] * ts + intercepts[k], ls=':', c="black", label='fit',)
        ax.set_title(g)
        ax.set_xlabel("time")
        ax.set_ylabel(r'$\hat{\log(O)}$')
        if not rvals is None:
            ax.text(ts[-2], means[k][0] * 0.9, r'R = ' + str(rvals[k].round(3)), fontsize=8,)
    return fig, axes

