import numpy as np
import matplotlib.pyplot as plt
from eolearn.core import EOPatch
import earthpy.plot as ep
from mpl_toolkits.axes_grid1 import make_axes_locatable


sentinel_2_true_color = [3, 2, 1]
sentinel_2_false_color = [7, 3, 2]


def plot_eopatch(
    eopatch: EOPatch, rgb_bands, feature, time_index=0, stretch=True, ax=None
):
    bands_at_timestamp = eopatch[feature][time_index, :, :, :]
    _, _, bands = bands_at_timestamp.shape
    single_bands = [bands_at_timestamp[:, :, x] for x in range(bands)]

    return ep.plot_rgb(
        np.stack(single_bands), rgb=rgb_bands, stretch=stretch, ax=ax
    )


def plot_ndarray_band(
    band_data, stretch=True, figsize=(10, 10), cmap="gray", colorbar=True
):
    vmin = None
    vmax = None
    if stretch:
        vmin, vmax = np.percentile(band_data, [5, 95])
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(band_data, cmap=cmap, vmin=vmin, vmax=vmax)

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        im = fig.colorbar(im, cax=cax, orientation="vertical")

    return im


def plot_single_band(
    eopatch: EOPatch,
    feature,
    band_index=0,
    time_index=0,
    stretch=True,
    figsize=(10, 10),
    cmap="gray",
    colorbar=True,
):
    if len(eopatch[feature].shape) > 3:
        band_data = eopatch[feature][time_index, :, :, band_index]
    else:
        band_data = eopatch[feature][:, :, band_index]

    return plot_ndarray_band(
        band_data,
        stretch=stretch,
        figsize=figsize,
        cmap=cmap,
        colorbar=colorbar,
    )
