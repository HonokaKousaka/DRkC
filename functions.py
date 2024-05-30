import numpy as np
import random

def distancePS(centerSet: np.ndarray, i: int, complete: np.ndarray) -> float:
    """
    Returns the distance between a certain point and a certain set.
    Args:
        centerSet (np.ndarray): A numpy array containing confirmed center indexes
        i (int): The index of any point
        complete (np.ndarray): An adjacency matrix of the dataset
    Returns:
        min_distance (float): The distance between point and center set
    """
    min_distance = float("inf")
    for center in centerSet:
        distance = complete[center][i]
        if (distance < min_distance):
            min_distance = distance
    
    return min_distance

def distance_lossMax(centerSet: np.ndarray, i: int, complete: np.ndarray) -> tuple[float, int]:
    """
    Returns the distance between a certain point and a certain set, exclusively designed for loss-maximizing deletion.
    Args:
        centerSet (np.ndarray): A numpy array containing confirmed center indexes
        i (int): The index of any point
        complete (np.ndarray): An adjacency matrix of the dataset
    Returns:
        min_distance (float): The distance between point and center set
        dis_center (int): The index of the center caused the distance
    """
    min_distance = float("inf")
    dis_center = None
    for center in centerSet:
        distance = complete[center][i]
        if (distance < min_distance):
            min_distance = distance
            dis_center = center
    
    return min_distance, dis_center

def GMM(points_index: np.ndarray, k: int, complete: np.ndarray) -> np.ndarray:
    """
    Returns indexes of k centers after running GMM Algorithm.
    Args: 
        points_index (np.ndarray): The indexes of data
        k (int): A decimal integer, the number of centers
        complete (np.ndarray): An adjacency matrix of the dataset
    Returns:
        centers (np.ndarray): A numpy array with k indexes as center point indexes
    """
    centers = []
    initial_point_index = random.choice(points_index)
    centers.append(initial_point_index)
    while (len(centers) < k):
        max_distance = 0
        max_distance_vector_index = None
        for i in points_index:
            distance = distancePS(centers, i, complete)
            if distance > max_distance:
                max_distance = distance
                max_distance_vector_index = i
        centers.append(max_distance_vector_index)
    centers = np.array(centers)

    return centers

def loss(centerSet: np.ndarray, points_index: np.ndarray, complete:np.ndarray) -> float:
    """
    Returns the loss value with centers and data points.
    Args:
        centerSet (np.ndarray): The numpy array containing all center indexes
        points_index (np.ndarray): The indexes of data
        complete (np.ndarray): An adjacency matrix of the dataset
    Returns:
        loss_value (float): The loss value of certain centers and data points
    """
    max_distance = float("-inf")
    for i in points_index:
        distance = distancePS(centerSet, i, complete)
        if (distance > max_distance):
            max_distance = distance
    
    return max_distance

def GB_deletion(z: int, complete: np.ndarray) -> np.ndarray:
    """
    Returns the left indexes and the deleted indexes after z-point deletion with GMM.
    Args: 
        z (int): A decimal integer, the number of deleted points
        complete (np.ndarray): An adjacency matrix of the dataset
    Returns:
        deleted_points (np.ndarray): A numpy array with n-z indexes
        deletion_points (np.ndarray): A numpy array with z indexes
    """
    amount = complete.shape[0]
    complete_array = np.arange(amount)
    deletion_points = GMM(complete_array, z, complete)
    deleted_points = np.setdiff1d(complete_array, deletion_points)

    return deleted_points, deletion_points

def k_NN(number_neighbors: int, points_index: np.ndarray, query_point: int, complete: np.ndarray) -> np.ndarray:
    """
    Returns the k nearest neighbors of the query points.
    Args:
        number_neighbors (np.ndarray): A demical integer, the number of neighbors
        points_index (np.ndarray): The indexes of data
        query_point (int): The index of any query point
        complete (np.ndarray): An adjacency matrix of the dataset
    Returns: 
        nearest_indices (np.ndarray): The indices of k nearest neighbors in points_index
    """
    distances = []
    for index in points_index:
        distances.append(complete[query_point][index])
    nearest_indices = np.argsort(distances)[:number_neighbors]

    return nearest_indices

def WBNN_deletion(coreSet: np.ndarray, z: int, complete: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the left indexes and the deleted indexes after deleting z clustered points.
    Args: 
        coreSet (np.ndarray): A numpy array as the coreset indexes
        z (int): A decimal integer, the number of deleted points
        complete (np.ndarray): An adjacency matrix of the dataset
    Returns:
        deleted_points (np.ndarray): A numpy array with n-z indexes
        deletion_points (np.ndarray): A numpy array with z indexes
    """
    amount = complete.shape[0]
    set_coreSet = set()
    for each in coreSet:
        set_coreSet = set_coreSet | each
    list_coreSet = list(set_coreSet)
    numpy_coreSet = np.array(list_coreSet)
    query_point = random.choice(numpy_coreSet)
    nearest_indices = k_NN(z, numpy_coreSet, query_point, complete)
    deletion_coreSet = numpy_coreSet[nearest_indices]
    complete_array = np.arange(amount)
    deleted_points = np.setdiff1d(complete_array, deletion_coreSet)
    
    return deleted_points, deletion_coreSet

def WBGreedy_deletion(coreSet: np.ndarray, z: int, complete: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the left indexes and the deleted indexes after deleting z points in a greedy way.
    Args: 
        coreSet (np.ndarray): A numpy array as the coreset indexes
        z (int): A decimal integer, the number of deleted points
        complete (np.ndarray): An adjacency matrix of the dataset
    Returns:
        deleted_points (np.ndarray): A numpy array with n-z indexes
        deletion_points (np.ndarray): A numpy array with z indexes
    """
    amount = complete.shape[0]
    set_coreSet = set()
    for each in coreSet:
        set_coreSet = set_coreSet | each
    list_coreSet = list(set_coreSet)
    numpy_coreSet = np.array(list_coreSet)
    complete_array = np.arange(amount)
    rest_points = np.setdiff1d(complete_array, numpy_coreSet)
    deletion_coreSet = []
    for i in range(z):
        deleted_point = None
        max_distance = float("-inf")
        for point in rest_points:
            possible_distance, possible_center = distance_lossMax(numpy_coreSet, point, complete)
            if possible_distance > max_distance:
                max_distance = possible_distance
                deleted_point = possible_center
        deletion_coreSet.append(deleted_point)
        indices = np.where(numpy_coreSet == deleted_point)[0]
        numpy_coreSet = np.delete(numpy_coreSet, indices)
    deletion_coreSet = np.array(deletion_coreSet)
    deleted_points = np.setdiff1d(complete_array, deletion_coreSet)
    
    return deleted_points, deletion_coreSet

def coreset_generate(points_index: np.ndarray,
                     centers_list: np.ndarray, 
                     z: int,
                     complete: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Returns the coreset indexes.
    Args: 
        points_index (np.ndarray): The indexes of data
        centers_list (np.ndarray): The center index set generated by GMM
        z (int): A decimal integer, the number of deleted points
        complete (np.ndarray): An adjacency matrix of the dataset
    Returns:
        coreset (np.ndarray): A numpy array as the coreset indexes
        center_mark (np.ndarray): A numpy array with the center index corresponding to each set in coreset
        size_coreset (int): A decimal integer, the size of coreset
    """
    coreset = []
    center_mark = []
    radius = loss(centers_list, points_index, complete)
    for i, center in enumerate(centers_list):
        center_mark.append(center)
        circles_index = []
        for k in points_index:
            if (complete[k][center] <= radius):
                circles_index.append(k)
        if (len(circles_index) <= z):
            coreset.append(set(circles_index))
        else:
            circle_index_array = np.array(circles_index)
            nearest_indices = k_NN(z+1, circle_index_array, center, complete)
            nearest_z_points = circle_index_array[nearest_indices]
            coreset.append(set(nearest_z_points))
    coreset = np.array(coreset)
    center_mark = np.array(center_mark)
    size_coreset = 0
    real_coreset = set()
    for each in coreset:
        real_coreset = real_coreset | each 
    size_coreset = len(real_coreset)

    return coreset, center_mark, size_coreset

def robust_solution(coreset: np.ndarray, 
                    center_mark: np.ndarray, 
                    deletion_points: np.ndarray, 
                    k: int) -> np.ndarray:
    """
    Returns the new center set index by our algorithm after deletion.
    Args: 
        coreset (np.ndarray): A numpy array with k sets
        center_mark (np.ndarray): A numpy array with the center index corresponding to each set in coreset
        deletion_points (np.ndarray): A numpy array with z already deleted point indexes
        k (int): A decimal integer, the number of centers
    Returns:
        new_centerSet (np.ndarray): A numpy array with k point indexes as center point indexes after deletion
    """
    new_centerSet = []
    new_coreset = set()
    set_deletion_points = set(deletion_points)
    for i in range(k):
        set_coreset = coreset[i]
        deleted_coreset = set_coreset - set_deletion_points
        new_coreset = new_coreset | deleted_coreset
        if (center_mark[i] in deletion_points):
            if (len(deleted_coreset) == 0):
                continue
            else:
                array_deleted_coreset = np.array(list(deleted_coreset))
                new_center = random.choice(array_deleted_coreset)
                new_centerSet.append(new_center)
        else:
            new_centerSet.append(center_mark[i])
    number_centers = len(new_centerSet)
    if (number_centers < k):
        array_new_coreset = np.array(list(new_coreset))
        random_integers = random.sample(range(len(array_new_coreset)), k - number_centers)
        for integer in random_integers:
            new_centerSet.append(array_new_coreset[integer])
    new_centerSet = np.array(new_centerSet)

    return new_centerSet

def G(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the adjacency matrix of n points and the list of edges.
    Args: 
        points (np.ndarray): The dataset, a numpy array with n points
    Returns:
        complete (np.ndarray): The adjacency matrix of n points
        edges (np.ndarray): The numpy array of edges
    """
    length = points.shape[0]
    complete = np.zeros((length, length))
    edges = []
    for k, point_k in enumerate(points):
        for j, point_j in enumerate(points):
            distance_kj = np.linalg.norm(point_k - point_j)
            complete[k][j] = distance_kj
            if (j > k):
                edges.append(distance_kj)
    edges.sort()
    edges = np.array(edges)
    return complete, edges

def G_i2(complete: np.ndarray, edges: np.ndarray, i: int) -> np.ndarray:
    """
    Returns the adjacency matrix of G_i^2.
    Args: 
        complete (np.ndarray): The adjacency matrix of n points
        edges (np.ndarray): The numpy array of edges
        i (int): A decimal integer, determining the longest edge of the graph
    Returns:
        matrix (np.ndarray): The adjacency matrix of G_i^2
    """
    G_i = np.copy(complete)
    length = complete.shape[0]
    threshold = edges[i]
    indices = complete > threshold
    G_i[indices] = 0
    for i in range(length):
        indices_i = G_i[i] > 0
        for j in range(i+1, length):
            indices_j = G_i[j] > 0
            indices_ij = indices_i & indices_j
            if True in indices_ij:
                G_i[i][j] = complete[i][j]
                G_i[j][i] = complete[j][i]
    
    matrix = G_i
    return matrix

def a_neighbor(matrix: np.ndarray, alpha: int) -> np.ndarray:
    """
    Returns several center indexes after running a_neighbor k-center algorithm.
    Args: 
        matrix (np.ndarray): The adjacency matrix of G_i^2
        alpha (int): A decimal integer, the number of nearest neighbors
    Returns:
        centers (np.ndarray): A set with some point indexes as center point indexes
    """
    centers = []
    length = matrix.shape[0]
    C_array = np.zeros(length)
    for j in range(alpha):
        for v in range(length):
            if C_array[v] < j:
                centers.append(v)
                C_array[v] = alpha
                for u in range(length):
                    if matrix[v][u] > 0:
                        C_array[u] += 1
    
    centers = np.array(centers)
    return centers

def a_neighbor_k_center(complete: np.ndarray, 
                        edges: np.ndarray, 
                        alpha: int, 
                        k: int) -> np.ndarray:
    """
    Returns the adjacency matrix of G_i^2.
    Args: 
        complete (np.ndarray): The adjacency matrix of n points
        edges (np.ndarray): The numpy array of edges
        alpha (int): A decimal integer, the number of nearest neighbors
        k (int): A decimal integer, the number of centers
    Returns:
        centers (np.ndarray): A numpy array with k point indexes as center point indexes
    """
    low = 0
    high = len(edges) - 1
    centers = None
    while low <= high:
        mid = (low + high) // 2
        matrix = G_i2(complete, edges, mid)
        centers = a_neighbor(matrix, alpha)
        number_centers = len(centers)
        if number_centers == k:
            return centers
        elif number_centers > k:
            low = mid + 1
        else:
            high = mid - 1
    if len(centers) < k:
        amount = complete.shape[0]
        numpy_centers = np.array(centers)
        complete_array = np.arange(amount)
        rest_points = np.setdiff1d(complete_array, numpy_centers)
        random_integers = random.sample(range(len(rest_points)), k - len(centers))
        for integer in random_integers:
            centers.append(rest_points[integer])
    centers = np.array(centers)

    return centers