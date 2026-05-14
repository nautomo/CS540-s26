import csv
from matplotlib import pyplot as plt
import numpy as np
from scipy.cluster import hierarchy

"""
Input: A string, representing the path to the file to be read.
Output: A list, where each element is a dictionary representing one row of the file.
"""
def load_data(filepath):
    data = []
    # Read in the file specified by filepath.
    with open(filepath, newline='', encoding='utf-8') as f:
        # Each row in the dataset is represented as a dictionary with the column headers as keys and 
        # the row elements as values
        reader = csv.DictReader(f)
        for row in reader:
            data.append(dict(row))
    # Return a list of dictionaries
    return data

"""
Input: A dictionary representing one country.
Output: A NumPy array of shape (9,) and dtype float64, the first element is x1 and so on with the 
        last element being x9.
"""
def calc_features(row):
    keys = ["child_mort", "exports", "health", "imports", "income", "inflation", "life_expec", 
            "total_fer", "gdpp"]
    return np.array([float(row[key]) for key in keys], dtype=np.float64)

"""
Input:
    features: A list of NumPy arrays of shape (9,), where each array is an (x1, ..., x9) feature 
              representation. The total number of feature vectors, i.e., the length of the input 
              list, is n.
    linkage_type: A string of the linkage type to use. Can be "complete" or "single". For any output
                  examples shown or described in following sections, assume the linkage type was 
                  "complete" unless otherwise specified.
Output: A NumPy array of shape (n - 1) * 4. For any i, Z[i, 0] and Z[i, 1] represent the indices of
        the two clusters that were merged in the ith iteration of the clustering algorithm. Then, 
        Z[i, 2] = d(Z[i, 0], Z[i, 1]) is the linkage distance between the two clusters that were 
        merged in the ith iteration (this will be a real value, not an integer like the other 
        quantities). Lastly, Z[i, 3] is the size of the new cluster formed by the merge, i.e., the 
        total number of countries in this cluster. Note that the original countries are considered 
        clusters indexed by 0, ..., n - 1, and the cluster constructed in the ith iteration (i >= 1) 
        of the algorithm has cluster index (n - 1) + i. Also, there is a tie-breaking rule specified 
        below that must be followed.
"""
def hac(features, linkage_type):
    # 1. Number each of your starting data points from 0 to n − 1. These are their original cluster numbers.
    n = len(features)
    # 2. Create an (n − 1) × 4 array or list. Iterate through this array/list row by row. For each row:
    Z = np.zeros((n-1, 4))
    distance_matrix = np.full((2*n-1, 2*n-1), np.inf)
    for i in range(n):
        for j in range(i+1, n):
            distance = np.linalg.norm(features[i] - features[j])
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance
        active_clusters = set(range(n))
    cluster_sizes = {i: 1 for i in range(n)}
    
    for i in range(n - 1):
        # (a) Determine which two clusters are closest and put their numbers into the first and second elements of the row, Z[i, 0] and Z[i, 1]. The first element listed, Z[i, 0], should be the smaller of the two cluster indexes.
        min_distance = np.inf
        best_pair = None
        
        for a in active_clusters:
            for b in active_clusters:
                if a < b:
                    d = distance_matrix[a][b]
                    if (best_pair is None or
                        d < min_distance or 
                        (d == min_distance and 
                         (a < best_pair[0] or (a == best_pair[0] and b < best_pair[1])))):
                        min_distance = d
                        best_pair = (a, b)
        
        a, b = best_pair
        new_cluster_index = n + i
        
        # (b) The linkage distance (complete or single) between the two clusters goes into the third element of the row, Z[i, 2].
        # (c) The total number of countries in the cluster goes into the fourth element, Z[i, 3].
        Z[i][0] = a
        Z[i][1] = b
        Z[i][2] = min_distance
        Z[i][3] = cluster_sizes[a] + cluster_sizes[b]
        
        # If you merge a cluster containing more than one country, its index (for the first or second element of
        # the row) is given by n+ the row index in which the cluster was created.
        for c in active_clusters:
            if c != a and c != b:
                if linkage_type == "single":
                    new_dist = min(distance_matrix[a][c],
                                   distance_matrix[b][c])
                else:  # complete
                    new_dist = max(distance_matrix[a][c],
                                   distance_matrix[b][c])

                distance_matrix[new_cluster_index][c] = new_dist
                distance_matrix[c][new_cluster_index] = new_dist

        active_clusters.remove(a)
        active_clusters.remove(b)
        active_clusters.add(new_cluster_index)
        cluster_sizes[new_cluster_index] = (cluster_sizes[a] + cluster_sizes[b])
    
    # 3. Before returning the data structure, convert it into a NumPy array if it isn’t one already.
    return Z

"""
Input: A NumPy array Z output from hac, and a list of string names corresponding to country names
       with length n.
Output: A matplotlib figure with a graph that visualizes hierarchical clustering.
"""
def fig_hac(Z, names):
    fig = plt.figure()
    hierarchy.dendrogram(Z, labels=names, leaf_rotation=90)
    fig.tight_layout()
    return fig

"""
Input: A list of the feature vectors output from calc_features. Each feature vector is a NumPy array
       with shape (9,) and dtype float64.
Output: A list of NumPy arrays with shape (9,) and dtype float64. However, the statistic values in 
        the feature vectors should be replaced with their normalized values.
"""
def normalize_features(features):
    # (9,) array -> (n, 9) matrix
    feature_matrix = np.vstack(features)

    means = np.mean(feature_matrix, axis=0)
    sds = np.std(feature_matrix, axis=0)

    normalized = (feature_matrix - means) / sds

    return [normalized[i].astype(np.float64) for i in range(len(features))]
"""
Input:
    features: A list of NumPy arrays of shape (9,), where each array is an (x1, ..., x9) feature 
              representation as computed in Section 4.2. The total number of feature vectors is n.
    k: An integer representing the number of clusters to create.
    max_iter: Maximum number of iterations (default=100).
    tol: Convergence tolerance (default=1e-4).
Output: A tuple containing four elements:
    1. labels: A NumPy array of shape (n,) containing cluster assignments for each data point 
               (values from 0 to k - 1).
    2. centers: A NumPy array of shape (k, 9) containing the final cluster centers.
    3. total_distortion: A float representing the sum of squared distances from each point to its 
                         assigned cluster center.
    4. num_iterations: An integer representing the number of iterations until convergence.
"""
def kmeans(features, k, max_iter=100, tol=1e-4):
    # 1. Initialization: Randomly select k data points from the input features as initial cluster centers.
    feature_matrix = np.vstack(features)
    indices = np.random.choice(feature_matrix.shape[0], k, replace=False)
    old_centers = feature_matrix[indices]
    
    for iteration in range(max_iter):
        # 2. Assignment Step: For each data point, assign it to the nearest cluster center using Euclidean distance.
        distances = np.linalg.norm(feature_matrix[:, None] - old_centers, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # 3. Update Step: For each cluster, compute the new center as the mean of all data points assigned to that cluster.
        new_centers = []
        for j in range(k):
            cluster_points = feature_matrix[labels == j]
            if len(cluster_points) == 0:
                new_center = old_centers[j]
            else:
                new_center = np.mean(cluster_points, axis=0)
            new_centers.append(new_center)
        new_centers = np.vstack(new_centers)
        
        # 4. Convergence Check: Compute the center shift. If center_shift < tol, the algorithm has converged. Otherwise, repeat steps 2-4.
        center_shift = np.linalg.norm(old_centers-new_centers)
        num_iterations = iteration + 1
        
        # 5. Termination: The algorithm stops when either:
        # The center shift is less than the tolerance (tol), or
        # The maximum number of iterations (max_iter) is reached.
        if center_shift < tol:
            break

        old_centers = new_centers

    # 6. Return Values: After convergence, compute and return:
    # Final cluster labels for all points
    # Final cluster centers
    # Total distortion
    # Number of iterations performed
    centers = old_centers

    total_distortion = 0.0
    for i in range(feature_matrix.shape[0]):
        total_distortion += np.linalg.norm(feature_matrix[i] - centers[labels[i]]) ** 2
    
    return (labels, centers, total_distortion, num_iterations)

if __name__ == "__main__":
    data = load_data("Country-data.csv")
    features = [calc_features(row) for row in data]
    names = [row["country"] for row in data]
    features_normalized = normalize_features(features)
    # Test HAC
    n = 50
    Z = hac(features_normalized[:n], linkage_type="complete")
    np.savetxt("output.txt", Z) # Compare with provided output.txt
    # Visualize dendrogram (use smaller n for readability)
    n = 20
    Z_small = hac(features_normalized[:n], linkage_type="complete")
    fig = fig_hac(Z_small, names[:n])
    plt.show()
    # Test k-means
    np.random.seed(42) # For reproducible results
    labels, centers, distortion, iters = kmeans(features_normalized[:n], k=5)
    print(f"K-means converged in {iters} iterations")
    print(f"Total distortion: {distortion}")
    print(f"Cluster labels: {labels}")