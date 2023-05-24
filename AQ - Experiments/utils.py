import haversine as hs
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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

# def nearest_neighbors_graph(df, neighbours):
