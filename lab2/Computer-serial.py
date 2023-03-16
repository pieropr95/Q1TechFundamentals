import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import copy
import seaborn as sns
from kneed import KneeLocator

def initialization(points, k):
    # We retrieve the number of points N and the number of dimensions D
    N, D = points.shape               
    # We choose randomly the first centroid
    centroids = [points[0]]             
    for number in range(k-1):
        # We sum distance of each point with every centroid. cdist is producing a matrix Nxk_i
        dists=np.sum(cdist(points,centroids), axis=1) 
        # We normalize the distances
        dists /= np.sum(dists)                    
        # We look for the index which attains the maximum sum of distances to the centroids
        new_centroid_idx, = [np.argmax(np.array(dists))]
        centroids += [points[new_centroid_idx]]
    return centroids

def centroid_assignation(points, k, max_iter=20): 
    #Initialization
    centroids = initialization(points, k)
    # We have already initialized our centroids, lets iterate them now
    iteration = 0
    prev_centroids = None
    # All centroids need to be equal to stop, or max_iter must be reached
    while np.not_equal(centroids, prev_centroids).any() and iteration < max_iter: 
        # Compute distance between each point and each centroid
        dists = cdist(points,centroids)
        # Get index of closest centroid for each point and store it in centroid_idx array
        centroid_idx = np.argmin(dists, axis=1)
        # Now we update the model
        prev_centroids=copy.copy(centroids)
        for i in range(k):
            centroids[i] = np.mean(points[centroid_idx == i], axis = 0)
        for i, centroid in enumerate(centroids):
            # Catch any np.nans, resulting from a centroid having no points
            if np.isnan(centroid).any():  
                centroids[i] = prev_centroids[i]
        iteration += 1
    # We compute WCSS
    loss=0.0
    dists = cdist(points, centroids)
    for i in range(points.shape[0]):
        loss += dists[i][centroid_idx[i]]**2
    return centroid_idx, np.array(centroids), loss

if __name__ == '__main__':
    
    # Read the file, transform boolean variables and normalize the dataset
    start0 = time.time()
    
    dataset = pd.read_csv('computers.csv', usecols = range(1,10), 
                          dtype={"price":int,"speed":int,"hd":int,"ram":int,
                                 "screen":int,"cores":int,"cd":str,"laptop":str,
                                 "trend":int})
    dataset['cd'] = dataset['cd'].apply(lambda x: 0 if x=='no' else 1)
    dataset['laptop'] = dataset['laptop'].apply(lambda x: 0 if x=='no' else 1)
    maxprice = max(dataset['price'])
    normalizeddata = dataset.div(dataset.max(axis=0), axis=1)
    points = normalizeddata.to_numpy()

    print("Execution time reading: " + str(time.time() - start0))
    
    # 1.- Construct the elbow graph and find the optimal clusters number (k)
    start1 = time.time()
    possible_k = list(range(2,13))
    l_wcss = []
    for k in possible_k:
        centroid_idx, centroids, wcss = centroid_assignation (points, k)
        l_wcss.append(wcss)
        
    # 2. Plots the results of the elbow graph. Choose optimum k
    elbowcurve = KneeLocator(possible_k, l_wcss, curve='convex', direction='decreasing')
    k = elbowcurve.knee # Optimal k
    print("Execution time elbow: " + str(time.time() - start1))
    plt.plot(possible_k, l_wcss)
    plt.vlines(k, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.show()
    
    # 3.- Clusters the data using the optimum value using kmeans.
    start = time.time()
    centroid_idx, centroids, loss = centroid_assignation (points, k)
   
    print("Execution time kmeans with optimal k: " + str(time.time() - start))
    
    # 4.- Plot the first 2 dimensions of the cluster
    colors = ["blue","green","magenta","yellow","red","cyan","black","brown","grey","orange", "purple"]

    ctr = np.vstack(centroids)

    for i in range(k):
        color = colors[i]
        plt.scatter([element[0] for element in points[centroid_idx==i]], [element[1] for element in points[centroid_idx==i]], color = color,s = 4) 

    plt.scatter(ctr[:,0], ctr[:,1], marker = 's', s=200, c=colors[0:k])
    plt.show()
    
    # 5.- Finds and print the cluster with the highest price average.
    lst_avg = [centroid[0] for centroid in centroids]
    indexmax = lst_avg.index(max(lst_avg))
    print("Cluster with highest price average: color " +str(colors[indexmax]))
    print("Higuest price average: "+ str(max(lst_avg)*maxprice))
    
    #6.- Prints a heatmap with matplotlib.pyplot for the clusters centroids.    
    df = pd.DataFrame(list(map(list, zip(*[centroids[idx] for idx in range(k)]))), index =["price","speed","hd","ram","screen","cores","cd","laptop","trend"])
    df.columns = [f'{(colors[col]).capitalize()} centroid' for col in range(k)]
    df = df.sort_values(by ='price', axis=1)
    p1 = sns.heatmap(df, cmap ="Blues")
    plt.xticks(rotation=45)
    plt.show()