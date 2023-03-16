import time
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
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
        centroid_idx = np.argmin(dists, axis=1)# Now we update the model
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
    return  (k, loss)

def updatecentroids_async (points, centroid_idx, i):
    return i, np.mean(points[centroid_idx == i], axis = 0)

def updatecentroids_sync (points, centroid_idx, i):
    return np.mean(points[centroid_idx == i], axis = 0)
    
def read(skipping, chunksize):
    df = pd.read_csv('computers.csv', skiprows = skipping, nrows=chunksize, usecols = range(1,10), names=['price','speed','hd','ram','screen','cores','cd','laptop','trend'])
    df['cd'] = df['cd'].apply(lambda x: 0 if x=='no' else 1)
    df['laptop'] = df['laptop'].apply(lambda x: 0 if x=='no' else 1)
    return df

def readprocess(skipping, chunksize, dataset):
    df = pd.read_csv('computers.csv', skiprows = skipping, nrows=chunksize, usecols = range(1,10), names=['price','speed','hd','ram','screen','cores','cd','laptop','trend'])
    df['cd'] = df['cd'].apply(lambda x: 0 if x=='no' else 1)
    df['laptop'] = df['laptop'].apply(lambda x: 0 if x=='no' else 1)
    dataset.append(df)

def collect_result (partialresult):
    global dataset
    dataset.append(partialresult)
    
def collect_wcss (partialresult):
    global l_wcss
    l_wcss.append(partialresult)
    
def collect_centroids_apply(tupla):
    global centroids
    centroids[tupla[0]] = tupla[1]
    
def collect_centroids_starmap(tupla):
    global lcentroids
    lcentroids.append(tupla)
    
if __name__ == '__main__':
    numprocesses = mp.cpu_count() # CHANGEABLE PARAMETER
    nchunks = 4 # CHANGEABLE PARAMETER
    chunksize = ceil(500000/nchunks) # CHANGEABLE PARAMETER
    l_skipping = [1+i*chunksize for i in range(nchunks)]
    
    # Read the file, transform boolean variables and normalize the dataset
    start0 = time.time()
    
    # OPTION 1: In serial (FASTEST for 500k data)
    dataset = pd.read_csv('computers.csv', usecols = range(1,10), dtype={"price":int,"speed":int,"hd":int,"ram":int,"screen":int,"cores":int,"cd":str,"laptop":str,"trend":int})
    dataset['cd'] = dataset['cd'].apply(lambda x: 0 if x=='no' else 1)
    dataset['laptop'] = dataset['laptop'].apply(lambda x: 0 if x=='no' else 1)
    
    
    # OPTION 2: apply 
    # p = mp.Pool(processes = numprocesses)
    # dataset = [p.apply (read_sync, args=(skipping, chunksize)) for skipping in l_skipping]
    # p.close()
    # dataset = pd.concat(dataset)
    
    # OPTION 3: starmap 
    # p = mp.Pool(processes = numprocesses)
    # dataset = p.starmap(read_sync, [(skipping, chunksize) for skipping in l_skipping])
    # p.close()
    # dataset = pd.concat(dataset)
    
    # OPTION 4: apply_async 
    # global dataset
    # dataset = []
    # p = mp.Pool(processes = numprocesses)
    # for skipping in l_skipping:
    #     p.apply_async(read, args=(skipping, chunksize), callback=collect_result) 
    # p.close()
    # p.join()
    # dataset = pd.concat(dataset)
    
    # OPTION 5: starmap_async 
    # global dataset
    # dataset = []
    # p = mp.Pool(processes = numprocesses)
    # p.starmap_async(read, [(skipping, chunksize) for skipping in l_skipping], callback=collect_result)
    # p.close()
    # p.join()
    # dataset = pd.concat(dataset[0])
    
    # OPTION 6: Process 
    # manager = mp.Manager()
    # dataset = manager.list()
    # processes = []
    # for skipping in l_skipping:
    #     process = mp.Process(target=readprocess, args=(skipping, chunksize,dataset,))
    #     processes.append(process)     
    # for process in processes:
    #     process.start()
    # for process in processes:
    #     process.join()
    # dataset = pd.concat(dataset)
    
    # COMMON TO ALL OPTIONS
    maxprice = max(dataset['price'])
    normalizeddata = dataset.div(dataset.max(axis=0), axis=1)
    points = normalizeddata.to_numpy()    
    
    print("Execution time reading: " + str(time.time() - start0))
    
    # 1.- Construct the elbow graph and find the optimal clusters number (k)
    start1 = time.time()
    possible_k = list(range(2,13))
    
    # OPTION 1: apply
    # p = mp.Pool(processes = numprocesses)
    # l_wcss = [p.apply(centroid_assignation, args= (points, k)) for k in possible_k]
    # p.close()
    
    # OPTION 2: starmap
    # p = mp.Pool(processes = numprocesses)
    # l_wcss = p.starmap(centroid_assignation, [(points, k) for k in possible_k])
    # p.close()
    
    # OPTION 3: apply_async 
    # global l_wcss
    # l_wcss = []
    # p = mp.Pool(processes = numprocesses)
    # for k in possible_k:
    #     p.apply_async(centroid_assignation, args=(points, k), callback=collect_wcss) 
    # p.close()
    # p.join()
    
    # OPTION 5: starmap_async
    global l_wcss
    l_wcss = []
    p = mp.Pool(processes = numprocesses)
    p.starmap_async(centroid_assignation, [(points, k) for k in possible_k], callback=collect_wcss)
    p.close()
    p.join()
    l_wcss = l_wcss[0]
    
    # 2. Plots the results of the elbow graph. Choose optimum k
    l_wcss = sorted(l_wcss, key = lambda x: x[0])
    wcss = [el[1] for el in l_wcss]
    elbowcurve = KneeLocator(possible_k, wcss, curve='convex', direction='decreasing')
    k = elbowcurve.knee # Optimal k
    print("Execution time elbow parallel: " + str(time.time() - start1))  
    plt.plot(possible_k, wcss)
    plt.vlines(k, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.show()
    
    # 3.- Clusters the data using the optimum value using kmeans.
    global centroids
    start = time.time()
    centroids = initialization(points, k)
    # We have already initialized our centroids, lets iterate them now
    iteration = 0
    prev_centroids = None
    max_iter = 20
    while np.not_equal(centroids, prev_centroids).any() and iteration < max_iter: #All centroids need to be equal to stop
        dists = cdist(points,centroids)
        centroid_idx = np.argmin(dists, axis=1)
        # Now we update the model
        prev_centroids = copy.copy(centroids)
        # OPTION 1: apply 
        # p = mp.Pool(processes = numprocesses)
        # centroids = [p.apply(updatecentroids_sync, args= (points, centroid_idx, i)) for i in range(k)]
        # p.close()
        
        # OPTION 2: starmap 
        # p = mp.Pool(processes = numprocesses)
        # centroids = p.starmap(updatecentroids_sync, [(points, centroid_idx, i) for i in range(k)])
        # p.close()
        
        # OPTION 3: apply_async 
        # p = mp.Pool(processes = numprocesses)
        # for i in range(k):
        #     p.apply_async(updatecentroids_async, args=(points, centroid_idx, i), callback=collect_centroids_apply) 
        # p.close()
        # p.join()
        # print(centroids)
        
        # OPTION 4: starmap_async - FASTEST
        global lcentroids
        lcentroids = []
        p = mp.Pool(processes = numprocesses)
        p.starmap_async(updatecentroids_async, [(points, centroid_idx, i) for i in range(k)], callback=collect_centroids_starmap)
        p.close()
        p.join()       
        for tupla in lcentroids[0]:
            centroids[tupla[0]] = tupla[1]
        
        # COMMON TO ALL OPTIONS
        for i, centroid in enumerate(centroids):
            if np.isnan(centroid).any():  
                # Catch any np.nans, resulting from a centroid having no points
                centroids[i] = prev_centroids[i]
        
        iteration += 1
        
    print("Execution time kmeans with optimal k: " + str(time.time() - start))
    
    # 4.- Plot the first 2 dimensions of the cluster 
    colors = ["blue","green","magenta","yellow","red","cyan","black","brown","grey","orange", "purple"]

    ctr = np.vstack(centroids)

    for i in range(k):
        color = colors[i]
        plt.scatter([element[0] for element in points[centroid_idx==i]], [element[1] for element in points[centroid_idx==i]], color = color,s = 3) 

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