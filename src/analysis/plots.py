import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr

HISTOGRAM_RESOLUTION = 0.05


def plot_olr_histogram(
    label,
    ds=None,
    lat_bounds=None,
    lon_bounds=None,
    hist=None,
    bin_edges=None,
    ax=None,
    olr_key="rlut",
    hist_range=(65.7, 399.85),
    hist_step=HISTOGRAM_RESOLUTION,
    color=(0.1, 0.1, 0.1),
    xmin=80,
    xmax=340,
):
    if bin_edges is None:
        bin_edges = np.arange(hist_range[0], hist_range[1] + hist_step / 2, hist_step)

    if hist is None:
        assert ds is not None and lat_bounds is not None and lon_bounds is not None
        # extract relevant OLR pixels
        mask = (
            (ds.lat >= lat_bounds[0])
            & (ds.lat <= lat_bounds[1])
            & (ds.lon >= lon_bounds[0])
            & (ds.lon <= lon_bounds[1])
        )
        ds_sub = ds.where(mask, drop=True)
        olr = ds_sub[olr_key].values

        hist, _ = np.histogram(olr, bins=bin_edges)

    normalised_hist = hist / hist.sum()

    # plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    ax.bar(
        bin_edges[:-1],
        normalised_hist * 100,
        width=hist_step,
        color=color,
        align="edge",
        alpha=0.1,
        label=label,
    )
    ax.step(
        bin_edges[:-1],
        normalised_hist * 100,
        color=color,
        linestyle="-",
        linewidth=1,
        where="post",
        alpha=0.8,
    )

    ax.set_xlim([xmin, xmax])

    ax.legend(loc="upper left")

    ax.set_xlabel("OLR (W/m²)")
    ax.set_ylabel("Pixel Counts (%)")
    ax.set_title("OLR Histograms")

    return hist, bin_edges


def plot_example_olr(ds, label, lat_bounds, lon_bounds, vmin=100, vmax=300):
    mask = (
        (ds.lat >= lat_bounds[0])
        & (ds.lat <= lat_bounds[1])
        & (ds.lon >= lon_bounds[0])
        & (ds.lon <= lon_bounds[1])
    )

    ds_sub = ds.where(mask, drop=True)

    llat = ds_sub.lat.values
    llon = ds_sub.lon.values
    rluut = ds_sub.rlut.values

    fig, ax = plt.subplots(
        1, 1, figsize=(7, 3), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    s = ax.scatter(llon, llat, c=rluut, s=0.03, cmap="Blues", vmin=vmin, vmax=vmax)
    ax.coastlines()

    ax.set_yticks([-10, 0, 10], crs=ccrs.PlateCarree())
    ax.set_yticklabels(["10°S", "0°", "10°N"])
    ax.tick_params(axis="y", length=0)

    ax.set_xticks([-100, -80, -60, -40], crs=ccrs.PlateCarree())
    ax.set_xticklabels(
        [
            "100°W",
            "80°W",
            "60°W",
            "40°W",
        ]
    )
    ax.tick_params(axis="x", length=0)

    t_str = str(ds_sub.time.values).split(".")[0].replace("T", " ")[:-3]
    ax.set_title(f"{label} {t_str}UTC")

    ax.set_extent([-120, -30, -20, 20], crs=ccrs.PlateCarree())

    # colorbar for the second row
    cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.75])  # tweak as needed
    fig.colorbar(s, cax=cbar_ax, label=r"OLR [$W/m^2$]", extend="both")

    return fig, ax


def plot_likelihood_hist(
    likelihoods_bpd,
    color,
    label=None,
    num_bins=80,
    fig=None,
    ax=None,
    linewidth=1,
    hatching=None,
    alpha=(1.0,),
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))

    n, bins, patches = ax.hist(
        likelihoods_bpd,
        bins=num_bins,
        density=True,
        label=label,
        color=color + (0.0,),
        edgecolor=color + alpha,
        histtype="stepfilled",
        linewidth=linewidth,
    )

    if hatching is not None:
        for patch in patches:
            patch.set_hatch("///")

    ax.set_xlabel("Log Likelihood (BPD)")
    ax.set_ylabel("Frequency")

    return fig, ax


def likelihood_scatter(
    x_variable, y_likelihoods, var_label, title="", marginals=True, color="Blue"
):
    fig = plt.figure(figsize=(5 * 1.5, 3.3 * 1.5))
    gs = GridSpec(
        2, 2, width_ratios=(4, 1), height_ratios=(1, 4), hspace=0.1, wspace=0.05
    )

    ax = fig.add_subplot(gs[1, 0])
    if marginals:
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

    # --- scatter ---
    ax.scatter(x_variable, y_likelihoods, s=3, marker="x", alpha=0.2, color=color)

    xx_txt = ax.get_xlim()[1] * 0.7
    yy_txt = ax.get_ylim()[1] * 0.9

    r_corr = pearsonr(x_variable, y_likelihoods)
    ax.text(xx_txt, yy_txt, rf"$r = {np.round(r_corr[0],2)}$")

    n_samples = len(x_variable)
    ax.text(xx_txt, yy_txt * 0.95, rf"$n = {n_samples}$", color=color)

    ax.set_xlabel(var_label)
    ax.set_ylabel("likelihood (bpd)")

    if marginals:
        n_bins = 50

        # ======================
        # Top marginal histogram
        # ======================
        ax_top.hist(
            x_variable,
            bins=n_bins,
            density=True,
            histtype="stepfilled",
            alpha=0.5,
            linewidth=1,
            color=color,
        )

        # =======================
        # Right marginal histogram
        # =======================
        ax_right.hist(
            y_likelihoods,
            bins=n_bins,
            density=True,
            orientation="horizontal",
            histtype="stepfilled",
            alpha=0.5,
            linewidth=1,
            color=color,
        )

        # --- clean up marginal axes ---

        ax_top.xaxis.set_visible(False)
        ax_top.spines["bottom"].set_visible(False)
        ax_top.spines["top"].set_visible(False)
        ax_top.spines["right"].set_visible(False)

        ax_top.set_ylabel("density")

        ax_right.yaxis.set_visible(False)
        ax_right.spines["left"].set_visible(False)
        ax_right.spines["bottom"].set_visible(False)
        ax_right.spines["right"].set_visible(False)

        ax_right.set_xlabel("density")
        ax_right.xaxis.set_label_position("top")

        ax_right.tick_params(
            axis="x", bottom=False, labelbottom=False, top=True, labeltop=True
        )

        ax_top.set_title(title)
    else:
        ax.set_title(title)
