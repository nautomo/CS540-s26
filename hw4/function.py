import geopandas
import numpy as np
from matplotlib import pyplot as plt

def worldmap_hac(Z, names, K_clusters):
    # Use 10m high-resolution dataset for complete coverage (167/167 countries)
    world = geopandas.read_file('https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip')

    # Use NAME_LONG field for best country name matching
    world['name'] = world['NAME_LONG'].str.strip()
    names = [name.strip() for name in names]

    world['cluster'] = np.nan

    n = len(names)
    clusters = {j: [j] for j in range(n)}

    for step in range(n-K_clusters):
        cluster1 = Z[step][0]
        cluster2 = Z[step][1]

        # Create new cluster id as n + step
        new_cluster_id = n + step

        # Merge clusters
        clusters[new_cluster_id] = clusters.pop(cluster1) + clusters.pop(cluster2)

    # Assign cluster labels to countries in the world dataset
    for i, value in enumerate(clusters.values()):
        for val in value:
            world.loc[world['name'] == names[val], 'cluster'] = i

    # Plot the map
    fig, ax = plt.subplots(figsize=(15, 10))
    world.plot(column='cluster', ax=ax, legend=True, cmap='tab10',
               missing_kwds={"color": "lightgrey", "label": "Other countries"})
    ax.set_title(f'Hierarchical Clustering (Complete Linkage)\nk={K_clusters} clusters', fontsize=16)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def kmeans_world_map(features, names, k, num_runs=10):
    """
    Run k-means multiple times and visualize best result on a world map.

    Parameters:
    -----------
    features : list of np.array
        Normalized feature vectors
    names : list of str
        Country names
    k : int
        Number of clusters
    num_runs : int
        Number of times to run k-means (default: 10)
    """
    # Run k-means multiple times to find best clustering
    best_distortion = float('inf')
    best_labels = None

    print(f"Running k-means {num_runs} times with k={k}...")
    for run in range(num_runs):
        np.random.seed(run)
        labels, centers, distortion, iters = kmeans(features, k=k)
        print(f"  Run {run+1}: distortion = {distortion:.4f}, iterations = {iters}")

        if distortion < best_distortion:
            best_distortion = distortion
            best_labels = labels

    print(f"\nBest distortion: {best_distortion:.4f}")
    print(f"Cluster sizes: {np.bincount(best_labels)}")

    # Use 10m high-resolution dataset for complete coverage (167/167 countries)
    world = geopandas.read_file('https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip')

    # Use NAME_LONG field for best country name matching
    world['name'] = world['NAME_LONG'].str.strip()
    names_clean = [name.strip() for name in names]

    world['cluster'] = np.nan

    # Assign cluster labels to countries
    for i, name in enumerate(names_clean):
        world.loc[world['name'] == name, 'cluster'] = best_labels[i]

    # Plot the map
    fig, ax = plt.subplots(figsize=(15, 10))
    world.plot(column='cluster', ax=ax, legend=True, cmap='tab10',
               missing_kwds={"color": "lightgrey", "label": "Other countries"})
    ax.set_title(f'K-Means Clustering (k={k})\nBest distortion: {best_distortion:.2f}', fontsize=16)
    ax.axis('off')
    plt.tight_layout()
    plt.show()