import pandas as pd
import networkx as nx
from sodapy import Socrata
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import copy 
from scipy.cluster.vq import kmeans2, whiten
import seaborn as sns

# create DiGraph using k-means clustering --> this time including destination edges! 
def createDiGraphK2(data, sample_size=None, k=40, iters=40, name='k2-taxi', save=False):
    # cluster data using k-means
    print("finding outliers")
    pvalid_idx = np.array(getOutliers(data.pickup_latitude.astype(float), data.pickup_longitude.astype(float), 1))
    dvalid_idx = np.array(getOutliers(data.dropoff_latitude.astype(float), data.dropoff_longitude.astype(float), 1))
    valid_idx = np.intersect1d(pvalid_idx, dvalid_idx)
    print("removing outliers")
    plat = np.array(data.pickup_latitude.astype(float))
    plat = plat[valid_idx]
    plon = np.array(data.pickup_longitude.astype(float))
    plon = plon[valid_idx]
    dlat = np.array(data.dropoff_latitude.astype(float))
    dlat = dlat[valid_idx]
    dlon = np.array(data.dropoff_longitude.astype(float))
    dlon = dlon[valid_idx]
    print("clustering with k-means")
    coordinates= np.array(list(zip(plat, plon)) + list(zip(dlat, dlon)))
    coordinates = coordinates
    centroids, labels = kmeans2(whiten(coordinates), k, iter = iters) 
    
    
    # create graph with k nodes
    print("creating graph now")
    G = nx.DiGraph(); 
    if sample_size is None: 
        sample_size = len(data)
    for trip_num in range(sample_size):
        
        # get k-means centroid from src and dest
        rand_idx = random.randint(0, len(plat) - 1)
        pla = plat[rand_idx]
        plo = plon[rand_idx]
        dla = dlat[rand_idx]
        dlo = dlon[rand_idx]
        src = labels[rand_idx]
        dest = labels[rand_idx + len(plat)]

        # increase edge weight from src -->  dest by 1 
        if not G.has_node(src):
            G.add_node(src, lat=0, lon=0)
        if not G.has_node(dest):
            G.add_node(dest, lat=0, lon=0)
        if not G.has_edge(src, dest):
            G.add_edge(src, dest, weight=0)
        G[src][dest]['weight'] += 1
        
        # keep a running average of lat/lon of each node
        G.node[src]['lat'] = (G.node[src]['lat'] +  pla) / 2 
        G.node[src]['lon'] =  (G.node[src]['lon'] +  plo) / 2
        G.node[dest]['lat'] = (G.node[dest]['lat'] +  dla) / 2 
        G.node[dest]['lon'] =  (G.node[dest]['lon'] +  dlo) / 2
        
        # save graph every 1000 nodes
        if (save):
	        if (trip_num > 2000 and trip_num % 2000 == 0):
	            nx.write_graphml(G, "graphs/" + name + ".graphml")
    return G 


# return indices of valid points
def getOutliers(x, y, outlierConstant):
    a = np.array(x)
    b = np.array(y)
    upper_quartile_a = np.percentile(a, 75)
    lower_quartile_a = np.percentile(a, 25)
    upper_quartile_b = np.percentile(b, 75)
    lower_quartile_b = np.percentile(b, 25)
    IQR_a = (upper_quartile_a - lower_quartile_a) * outlierConstant
    IQR_b = (upper_quartile_b - lower_quartile_b) * outlierConstant
    quartileSet_a = (lower_quartile_a - IQR_a, upper_quartile_a + IQR_a)
    quartileSet_b = (lower_quartile_b - IQR_b, upper_quartile_b + IQR_b)
    valid = []
    for i,v in enumerate(a.tolist()):
        if (v >= quartileSet_a[0] and v <= quartileSet_a[1] and b[i] >= quartileSet_b[0] and b[i] <= quartileSet_b[1]):
            valid.append(i)
    return valid

def get_stats(G):
    stats = {};  
    stats['out_degrees'] = G.out_degree(weight='weight'); 
    stats['closeness_centrality'] = nx.closeness_centrality(G); # hard to do the rest of these because of uniform dist. 
    stats['betweenness_centrality'] = nx.betweenness_centrality(G); 
    stats['eigenvalue_centrality'] = nx.eigenvector_centrality(G); 
    stats['pagerank'] = nx.pagerank(G); 
    return stats