import numpy as np
import pandas as pd

from scipy.interpolate import griddata
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import autoroot
from notebooks.utils.patch_likelihoods import load_likelihood_dfs, load_all_dataset_likelihoods

from src.analysis import symmetric_kl, get_patch_likelihood_distribution, split_likelihoods


import seaborn as sns

IFS_COLOR = sns.color_palette("flare", n_colors=10)[5]
ICON_COLOR = sns.color_palette("mako", n_colors=10)[4]
GOES_COLOR = sns.color_palette("Greys", n_colors=15)[10]


def plot_likelihood_histograms(
    likelihood_dfs,
    land_ocean_dict,
    n_bins = 100
):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    ax = axs[0]
    
    train_thickness = 1.5
    val_thickness = 0.9
    
    for key, df in likelihood_dfs.items():
        if 'goes' in key: 
            color, zorder = GOES_COLOR, 2
        elif 'ifs' in key: 
            color, zorder = IFS_COLOR, 1
        elif 'icon' in key: 
            color, zorder = ICON_COLOR, 1
    
        label = key.split('_')[0].upper()
    
        n, bins, patches = ax.hist(
            df['bpd'],
            bins=n_bins,
            label=label,
            color=color + (0.0,),
            edgecolor=mcolors.to_hex(color),
            histtype='stepfilled',
            zorder=zorder,
            linewidth=train_thickness if 'train' in key else val_thickness,
            density=True,
        )
    
    # first legend: models
    legend1 = ax.legend(loc="upper left")
    ax.add_artist(legend1)
    
    
    ax.set_xlabel("Log Likelihood (BPD)")
    ax.set_xlim([1.5,10.5])
    ax.set_ylabel("Frequency")
    
    ax = axs[1]
    
    for key, likelihood_array in land_ocean_dict.items():
        if 'goes' in key: 
            color = GOES_COLOR
            zorder = 2
        elif 'ifs' in key: 
            color = IFS_COLOR
            zorder = 1
        elif 'icon' in key: 
            color = ICON_COLOR
            zorder = 1
        if 'ocean' in key: 
            alpha = (0.2,)
        else: 
            alpha = (0.2,)
    
        n, bins, patches = ax.hist(
            likelihood_array,
            bins=n_bins,
            color=color + (0.0,) if 'ocean' in key else color + (0.2,),
            edgecolor=mcolors.to_hex(color),
            histtype='stepfilled',
            zorder=zorder,
            label=key.split('_')[0].upper() if 'land' in key else None,
            linewidth=1.2,
            density=True,
        )
    
        # add hatching for ocean 
        if 'ocean' in key:
            for patch in patches:
                patch.set_hatch('///')
    
    
    # second legend: land vs ocean styles
    legend_elements = [
        Patch(facecolor=(0,0,0,.2), edgecolor="black", label="Land"),
        Patch(facecolor="white", hatch="///", edgecolor="black", label="Ocean"),
    ]
    legend2 = ax.legend(handles=legend_elements, loc="upper right", frameon=False, fontsize=9)
    
    plt.xlabel("Log Likelihood (BPD)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    
    # increase space between subplots
    plt.subplots_adjust(wspace=0.3)
    return fig, ax




def plot_d_skl_maps(
    all_dataset_likelihoods, 
    likelihood_vmin = 3.5,
    likelihood_vmax = 9,
    d_skl_vmin=0, 
    d_skl_vmax=10,
    skl_bins=10,
):
    
    fig, axs = plt.subplots(2, 3, figsize=(11.75, 3.5), subplot_kw={'projection': ccrs.PlateCarree()})

    patch_likelihoods = {}
    
    
    # define grid boundaries and resolution
    lon_min, lon_max, lat_min, lat_max = -120, -30, -20, 20
    res = 0.25
    
    grid_lon = np.arange(lon_min, lon_max + res, res)
    grid_lat = np.arange(lat_min, lat_max + res, res)
    lon_grid, lat_grid = np.meshgrid(grid_lon, grid_lat)
    
    for i, dataset in enumerate(['goes', 'ifs', 'icon']):
        df = all_dataset_likelihoods[dataset]
    
        dataset_patch_likelihoods = get_patch_likelihood_distribution(df, column="bpd")
        patch_likelihoods[dataset] = dataset_patch_likelihoods # store to use later for KL divergence
        points = dataset_patch_likelihoods[['lon', 'lat']].values
        mean_likelihood = dataset_patch_likelihoods['mean'].values
    
        grid_mean_likelihood = griddata(points, mean_likelihood, (lon_grid, lat_grid), method='nearest')
    
        ax = axs[0, i]
    
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
        if i == 0:
            ax.set_yticks([-10, 0, 10], crs=ccrs.PlateCarree())
            ax.set_yticklabels(['10°S', '0°', '10°N'])
            ax.tick_params(axis='y', length=0)
    
            ax.set_xticks([-100, -80, -60, -40], crs=ccrs.PlateCarree())
            ax.set_xticklabels(['100°W', '80°W', '60°W', '40°W', ])
            ax.tick_params(axis='x', length=0) 
        # Filled plot
        c = ax.pcolormesh(lon_grid, lat_grid, grid_mean_likelihood, cmap="viridis", shading='auto', vmin=likelihood_vmin, vmax=likelihood_vmax)
    
        ax.set_title(f"{dataset.upper()}")
    
    fig.subplots_adjust(wspace=0.01, hspace=0.05)
    
    # add a single colorbar for the second row
    cbar_ax = fig.add_axes([0.91, 0.5, 0.007, 0.35])
    cbar=fig.colorbar(c, cax=cbar_ax, label="Log-Likelihood (BPD)", extend='both')
    
    kl_dataframes = {}
    
    for i, dataset in enumerate(['goes', 'ifs', 'icon']):
        if dataset == 'goes':
            # remove subplot
            axs[1, i].axis('off')
            continue
    
        # get symmetric kl divergence between goes and ifs/icon for each patch
        kl_divs = []
        goes_patch_likelihoods = patch_likelihoods['goes']
        dataset_patch_likelihoods = patch_likelihoods[dataset]
    
        # build KDTree for fast nearest-neighbor search
        other_points = dataset_patch_likelihoods[['lon', 'lat']].values
        tree = cKDTree(other_points)
    
        for _, row in goes_patch_likelihoods.iterrows():
            lon, lat = row['lon'], row['lat']
            goes_values = row['values']
    
            # query nearest patch
            dist, idx = tree.query([lon, lat], k=1)
            if np.isfinite(dist):
                other_values = dataset_patch_likelihoods.iloc[idx]['values']
                kl_div = symmetric_kl(goes_values, other_values, bins=skl_bins)
                kl_divs.append((lon, lat, kl_div))
            else:
                kl_divs.append((lon, lat, np.nan)) # no match found
    
        kl_df = pd.DataFrame(kl_divs, columns=['lon', 'lat', 'kl_divergence'])
        kl_dataframes[dataset] = kl_df
    
        # interpolate to grid
        points = kl_df[['lon', 'lat']].values
        kl_values = kl_df['kl_divergence'].values
        grid_kl_divergence = griddata(points, kl_values, (lon_grid, lat_grid), method='nearest')
    
        ax = axs[1, i]
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
    
        if i == 1:
            ax.set_yticks([-10, 0, 10], crs=ccrs.PlateCarree())
            ax.set_yticklabels(['10°S', '0°', '10°N'])
            ax.tick_params(axis='y', length=0)
    
        ax.set_xticks([-100, -80, -60, -40], crs=ccrs.PlateCarree())
        ax.set_xticklabels(['100°W', '80°W', '60°W', '40°W', ])
        ax.tick_params(axis='x', length=0) 
    
        c = ax.pcolormesh(lon_grid, lat_grid, grid_kl_divergence, cmap="magma", shading='auto', vmin=0, vmax=10)
    
    # adjust spacing between subplots
    fig.subplots_adjust(wspace=0.01, hspace=0.05)
    
    # colorbar for the second row
    cbar_ax = fig.add_axes([0.91, 0.12, 0.007, 0.35])  # tweak as needed
    fig.colorbar(c, cax=cbar_ax, label=r"$D_{SKL}$", extend='max')

    return fig, ax