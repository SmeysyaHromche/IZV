#!/usr/bin/python3.10
# coding=utf-8
# %%%
import pandas as pd
import geopandas
import matplotlib
import matplotlib.pyplot as plt
import contextily as ctx
from sklearn.cluster import AgglomerativeClustering
import numpy as np

matplotlib.use('TkAgg')

def make_geo(df_accidents: pd.DataFrame, df_locations: pd.DataFrame) -> geopandas.GeoDataFrame:
    """
    Match accident data with geo data and create new data frame.
    Args:
        df_accidents : data about accidents
        df_locations : data about locations

    Returns:
        Accidents data with geo information
    """
    df = pd.merge(df_accidents, df_locations, on='p1')  # merge accidents and geo data to one frame
    df = df[(df['d'].notna()) & (df['e'].notna()) & (df['d'] != 0) & (df['e'] != 0)]  # clean d and e col
    df['d'], df['e'] = np.where(df['d'] > df['e'],
                                (df['d'].values, df['e'].values),
                                (df['e'].values, df['d'].values))  # check swapped d and e data
    return geopandas.GeoDataFrame(df,
                                  geometry=geopandas.points_from_xy(df['d'], df['e']),
                                  crs='EPSG:5514')

def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None,
             show_figure: bool = False):
    """
    Create accident by alcohol charts in geo representation and save it on target file with optional displaying.
    Args:
        gdf: accident and geo data
        fig_location: path/name for target file for saving. Defaults to None.
        show_figure: flag for optional displaying. Defaults to False.

    Returns:
        None
    """
    # aux const
    region = 'JHM'
    month = {1:'Leden', 2:'Unor'}

    # data preparing
    df = gdf[gdf['region'] == region]  # filter by trg region
    df = df[df['p11'] >= 4]  # filter by alcohol
    df = df[(df['date'].dt.month == 1) | (df['date'].dt.month == 2)]  # filter by trg month
    (minx, miny, maxx, maxy) = df.total_bounds

    # plotting
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 12))
    plt.tight_layout()
    for i, k in enumerate(month.keys()):
        df[df['date'].dt.month == k].plot(ax=ax[i], color='red', markersize=5, alpha=0.6)
        ax[i].set_xlim(xmin=minx, xmax=maxx)
        ax[i].set_ylim(ymin=miny, ymax=maxy)
        ax[i].set_title(f'{region} kraj - pod vlivem alkoholu ({month[k]})')
        ctx.add_basemap(ax[i], crs=df.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.9, zoom=10)
        ax[i].set_axis_off()

    # save and display
    if fig_location:
        plt.savefig(fig_location)
    if show_figure:
        plt.show()


def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """
    Create accident by animals charts in geo representation with Agglomerative clustering and save it on target file
    with optional displaying.

    Args:
        gdf: accident and geo data
        fig_location: path/name for target file for saving. Defaults to None.
        show_figure: flag for optional displaying. Defaults to False.

    Returns:
        None
    """
    # aux
    region = 'JHM'

    # data preparing
    df = gdf[gdf['region'] == region]  # filter by trg region
    df = df[df['p10'] == 4]  # filter by animals
    (minx, miny, maxx, maxy) = df.total_bounds

    # Clustering
    # ==========
    # I didn't want to limit myself to K-means clustering, so I decided to try other approaches.
    # The first one I tried was DBSCAN, but I had problems with data normalisation, esp selection and uneven data density.
    # Based on this, I make choose on Agglomerative clustering, it showed good test data distributions in my opinion.
    # Through experimentation, I found an average number of clusters that would divide the plane logically enough
    # without splitting it into smaller parts. I choose for linkage method 'ward' for better data compatibility.
    # ===========
    df['cluster'] = AgglomerativeClustering(n_clusters=12, linkage='ward').fit_predict(df[['d', 'e']])
    areas = df.groupby('cluster')['geometry'].apply(
        lambda x: x.unary_union.convex_hull).reset_index()
    polygons = geopandas.GeoDataFrame(geometry=areas['geometry'], crs=df.crs)
    polygons['accident_count'] = polygons.apply(
        lambda x: df[df.geometry.within(x.geometry)].shape[0], axis=1)

    # plotting
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.tight_layout()
    ax.set_xlim(xmin=minx, xmax=maxx)
    ax.set_ylim(ymin=miny, ymax=maxy)
    polygons.plot(ax=ax, column='accident_count', cmap='plasma', alpha=0.3)
    df.plot(ax=ax, color='red', markersize=5, alpha=0.6)

    # create colorbar based on number of accident in every cluster region
    norm = matplotlib.colors.Normalize(vmin=polygons['accident_count'].min(), vmax=polygons['accident_count'].max())
    cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap='plasma')
    plt.colorbar(cmap, ax=ax, orientation='horizontal', pad=0.02, label='Počet nehod v úseku')

    # map
    ctx.add_basemap(ax, crs=df.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.9, zoom=10)
    ax.set_axis_off()
    ax.set_title(f'Nehody v {region} kraji zaviněné lesní zvěří')

    if fig_location:
        plt.savefig(fig_location)
    if show_figure:
        plt.show()

if __name__ == "__main__":
    # zde muzete delat libovolne modifikace
    df_accidents = pd.read_pickle("accidents.pkl.gz")
    df_locations = pd.read_pickle("locations.pkl.gz")
    gdf = make_geo(df_accidents, df_locations)
    plot_geo(gdf, "geo1.png", True)
    plot_cluster(gdf, "geo2.png", True)