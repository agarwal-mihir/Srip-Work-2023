import haversine as hs
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import geopandas as gpd
import xarray as xr

def get_distance(lat1, lon1, lat2, lon2):
    return hs.haversine((lat1, lon1), (lat2, lon2))


def draw_graph(G, title):
    layout1 = nx.spring_layout(G, k=0.5)
    plt.figure()
    plt.title(title)
    nx.draw(G, pos=layout1, with_labels=True)
    plt.show()


def distance_threshold_graph(df, distance_threshold):
    G1 = nx.Graph()
    for i in range(len(df)):
        lat1, lon1 = df.iloc[i]["latitude"], df.iloc[i]["longitude"]
        pm = df.iloc[i]["PM2.5"]
        G1.add_node(i, latitude=lat1, longitude=lon1, pm=pm)
        for j in range(i + 1, len(df)):
            lat2, lon2 = df.iloc[j]["latitude"], df.iloc[j]["longitude"]
            distance = get_distance(lat1, lon1, lat2, lon2)
            if distance <= distance_threshold:
                G1.add_edge(i, j)
    return G1

def nearest_neighbors_graph(df, no_of_neighbours):
    
    le = LabelEncoder()
    df['station'] = le.fit_transform(df['station'])
    station = {i: [df[df['station'] == i]['latitude'].item(), df[df['station'] == i]['longitude'].item(), df[df['station'] == i]['PM2.5'].item()] for i in df.station.unique()}
    distances = []
    for i in station.keys():
        temp = []
        for j in station.keys():
            if i == j:
                continue
            temp.append([get_distance(station[i][0], station[i][1], station[j][0], station[j][1]), j])
        temp.sort()
        distances.append(temp)
    G = nx.Graph()

    for i, dist in enumerate(distances):
        G.add_node(i, latitude=station[i][0], longitude=station[i][1], pm=station[i][2])
        for j in range(no_of_neighbours):
            s = dist[j][1]
            G.add_node(s, latitude=station[s][0], longitude=station[s][1], pm=station[s][2])
            G.add_edge(i, s)
    return G

def plot_heatmap(df, lat, lon, values): 

    delhi_shapefile = gpd.read_file('data/Delhi/Districts.shp')
    gdf_data = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
    latitudes = np.array(df['latitude'])
    longitudes = np.array(df['longitude'])
    g_lat = np.linspace(latitudes.min()-0.1, latitudes.max()+0.1, 30)
    g_long = np.linspace(longitudes.min()-0.1, longitudes.max()+0.1, 30)
    lat_grid, lon_grid = np.meshgrid(g_lat, g_long)
    temp_data = gpd.GeoDataFrame(geometry = gpd.points_from_xy(lon_grid.flatten(), lat_grid.flatten()))
    delhi_shapefile = gpd.read_file('data/Delhi/Districts.shp')
    shapefile_extent = delhi_shapefile.total_bounds
    fig, ax = plt.subplots(figsize=(10, 10))
    contour = ax.contourf(lon.reshape(lon_grid.shape), lat.reshape(lon_grid.shape), values.reshape(lon_grid.shape), cmap='coolwarm', levels = 200)
    delhi_shapefile.plot(ax=ax, edgecolor='black', facecolor='none')
    gdf_data.plot(ax=ax, color='black', markersize=20, label='Air Stations')
    plt.colorbar(contour, label='PM2.5',shrink=0.7)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('PM2.5 Predictions Heatmap')
    plt.legend()
    plt.show()


    
