import haversine as hs
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import geopandas as gpd
import xarray as xr
import torch
from torch_geometric.data import Data, Dataset

def get_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two coordinates using the Haversine formula.

    Parameters:
    - lat1 (float): Latitude of the first point.
    - lon1 (float): Longitude of the first point.
    - lat2 (float): Latitude of the second point.
    - lon2 (float): Longitude of the second point.

    Returns:
    - distance (float): The distance between the two coordinates in kilometers.
    """
    return hs.haversine((lat1, lon1), (lat2, lon2))


def draw_graph(graph, title):
    """
    Draw a graph using the Spring layout algorithm and display it.

    Parameters:
    - graph (NetworkX graph): The graph to be drawn.
    - title (str): The title of the graph.

    Returns:
    - None
    """
    # Set the layout algorithm for node positioning
    layout = nx.spring_layout(graph, k=0.5)

    # Create a new figure
    plt.figure()

    # Set the title of the graph
    plt.title(title)

    # Draw the graph with labels
    nx.draw(graph, pos=layout, with_labels=True)

    # Show the graph
    plt.show()


def distance_threshold_graph(df, distance_threshold):
    """
    Create a graph based on a distance threshold using the given dataframe.

    Parameters:
    - df (pandas DataFrame): The dataframe containing latitude, longitude, and PM2.5 values.
    - distance_threshold (float): The maximum distance threshold for creating edges between nodes.

    Returns:
    - G (NetworkX graph): The resulting graph.
    """
    # Create an empty graph
    G = nx.Graph()

    # Iterate over each row in the dataframe
    for i in range(len(df)):
        lat1, lon1 = df.iloc[i]["latitude"], df.iloc[i]["longitude"]
        ws = df.iloc[i]['WS']
        wd = df.iloc[i]['WD']
        pm = df.iloc[i]["PM2.5"]
        station = df.iloc[i]["station"]
        
        # Add a node to the graph with latitude, longitude, PM2.5, and station attributes
        G.add_node(i, latitude=lat1, longitude=lon1, pm=pm, station=station, ws = ws, wd = wd)
        
        # Iterate over the remaining rows to check for edges
        for j in range(i + 1, len(df)):
            lat2, lon2 = df.iloc[j]["latitude"], df.iloc[j]["longitude"]
            
            # Calculate the distance between the two nodes
            distance = get_distance(lat1, lon1, lat2, lon2)
            
            # Check if the distance is within the threshold
            if distance <= distance_threshold:
                # Add an edge between the nodes
                G.add_edge(i, j)
    
    return G

def nearest_neighbors_graph(df, no_of_neighbours):
    """
    Create a graph based on the nearest neighbors using the given dataframe.

    Parameters:
    - df (pandas DataFrame): The dataframe containing latitude, longitude, and PM2.5 values.
    - no_of_neighbours (int): The number of nearest neighbors to consider for each node.

    Returns:
    - G (NetworkX graph): The resulting graph.
    """
    le = LabelEncoder()
    df['station_code'] = le.fit_transform(df['station'])
    station = {i: [df[df['station_code'] == i]['latitude'].item(), df[df['station_code'] == i]['longitude'].item(),
                   df[df['station_code'] == i]['PM2.5'].item(), df[df['station_code'] == i]['station'].item(),
                   df[df['station_code'] == i]['WS'].item(), df[df['station_code'] == i]['WD'].item()]
               for i in df.station_code.unique()}
    distances = []
    
    # Calculate distances between stations
    for i in station.keys():
        temp = []
        for j in station.keys():
            if i == j:
                continue
            temp.append([get_distance(station[i][0], station[i][1], station[j][0], station[j][1]), j])
        temp.sort()
        distances.append(temp)
    
    # Create an empty graph
    G = nx.Graph()

    # Add nodes and edges based on nearest neighbors
    for i, dist in enumerate(distances):
        G.add_node(i, latitude=station[i][0], longitude=station[i][1], pm=station[i][2], station=station[i][3], ws = station[i][4], wd = station[i][5])
        for j in range(no_of_neighbours):
            s = dist[j][1]
            G.add_node(s, latitude=station[s][0], longitude=station[s][1], pm=station[s][2], station=station[s][3], ws = station[i][4], wd = station[i][5])
            G.add_edge(i, s)
    
    return G

def plot_heatmap(df, latitudes, longitudes, values):
    """
    Plot a heatmap of PM2.5 predictions on a map.

    Parameters:
    - df (pandas DataFrame): The dataframe containing latitude, longitude, and PM2.5 values.
    - latitudes (numpy array): Array of latitude values.
    - longitudes (numpy array): Array of longitude values.
    - values (numpy array): Array of PM2.5 prediction values.

    Returns:
    - None (displays the plot)
    """

    # Read shapefile data
    delhi_shapefile = gpd.read_file('data/Delhi/Districts.shp')

    # Create GeoDataFrame from dataframe and points
    gdf_data = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))

    # Define grid for contour plot
    g_lat = np.linspace(latitudes.min() - 0.1, latitudes.max() + 0.1, 30)
    g_long = np.linspace(longitudes.min() - 0.1, longitudes.max() + 0.1, 30)
    lat_grid, lon_grid = np.meshgrid(g_lat, g_long)

    # Create temporary GeoDataFrame for grid points
    temp_data = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon_grid.flatten(), lat_grid.flatten()))

    # Read shapefile data
    delhi_shapefile = gpd.read_file('data/Delhi/Districts.shp')
    shapefile_extent = delhi_shapefile.total_bounds

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))

    # Calculate vmin and vmax based on percentiles
    vmin = np.percentile(values, 2)
    vmax = np.percentile(values, 98)

    # Create contour plot
    contour = ax.contourf(longitudes.reshape(lon_grid.shape), latitudes.reshape(lon_grid.shape), values.reshape(lon_grid.shape),
                          cmap='coolwarm', levels=200, vmin=vmin, vmax=vmax)

    # Plot shapefile boundaries
    delhi_shapefile.plot(ax=ax, edgecolor='black', facecolor='none')

    # Plot data points as black markers
    gdf_data.plot(ax=ax, color='black', markersize=20, label='Air Stations')

    # Plot data points as colored bubbles
    scatter = ax.scatter(df['longitude'], df['latitude'], s=df["PM2.5"], c=df["PM2.5"])
    
    df['wind_direction'] = 270 - df['WD']
    u = np.cos(np.radians(df['wind_direction']))
    v = np.sin(np.radians(df['wind_direction']))
    u1 = df['WS'] * np.cos(np.radians(df['wind_direction']))
    v1 = df['WS'] * np.sin(np.radians(df['wind_direction']))
    u1_ = u1.sum()
    v1_ = v1.sum()
    
#     ax.text(0.02, 0.02, "Net Wind")
      
    quiver = ax.quiver(df['longitude'], df['latitude'], u, v, alpha = 0.5, label = 'Wind')
    overall_quiver = ax.quiver(76.9, 28.85, u1_, v1_)
    plt.text(76.87, 28.88, "Overall Wind")
    
    # Add colorbars
    cbar1 = plt.colorbar(contour, label='PM2.5 - for contour plot', shrink=0.7)
    cbar2 = plt.colorbar(scatter, label='PM2.5 - for bubble plot', shrink=0.7)
    cbar1.ax.set_ylabel('PM2.5')
    cbar2.ax.set_ylabel('PM2.5')

    # Set x-axis and y-axis labels
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Set title and legend
    plt.title('PM2.5 Predictions Heatmap')
    plt.legend()

    # Display the plot
    plt.show()


def dataset_generation(G):
    """
    Generate the dataset for graph neural network training.

    Parameters:
    - G (NetworkX Graph): The input graph.

    Returns:
    - node_features (list): List of node features (latitude, longitude) for each node in the graph.
    - edge_index (torch.tensor): Edge index tensor representing the graph edges.
    - y (torch.tensor): Target tensor containing PM2.5 values for each node in the graph.
    """

    # Extract node features from graph
    node_features = [(G.nodes[node]['latitude'], G.nodes[node]['longitude'], G.nodes[node]['ws'], G.nodes[node]['wd']) for node in G.nodes]

    # Create edge index tensor
    undirected_edges = []
    for edge in G.edges:
        undirected_edges.append(edge)
        undirected_edges.append((edge[1], edge[0]))  # Add the reverse edge

    edge_index = torch.tensor(undirected_edges).t().contiguous()

    # Create target tensor
    y = torch.tensor([G.nodes[node]['pm'] for node in G.nodes], dtype=torch.float).view(-1, 1)

    return node_features, edge_index, y
