#import knnmeans as point
from numpy.lib.function_base import _ARGUMENT_LIST
from sklearn.neighbors import NearestNeighbors
from knnmeans import Point 
from builtins import type
#from clyent import color
from scipy.spatial import distance
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cmx
from random import randint
from random import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
import warnings
import numpy as np
import pandas as pd
from sklearn import datasets
#import SIKCHELP

if __name__=="__main__":



    #parameter
    k=5
    epsilon=0.1
    tmax=15

    np.warnings.filterwarnings("ignore", category=DeprecationWarning)
    # file einlesen

    labeled=False
    pointsHaveNames=False
    #filename='examples/'
    #Synthetic data
    #filename+='Compound.txt'; X=np.genfromtxt(filename); X=X[:,:2]
    #filename+='t4.8k.csv'; X=np.genfromtxt(filename); X=X[:,:2]
    #filename+='preswissroll.txt'; X=np.genfromtxt(filename); X=X[:,:2]
    #filename+='toy.txt'; X=np.genfromtxt(filename); X=X[:,:2]
    #filename+='pathbased.txt';X=np.genfromtxt(filename); X=X[:,:2]
    #filename+='mouse.txt'; X=np.genfromtxt(filename);  labeled=True;
    #filename+='twoMoonsStatic.csv'; X=np.genfromtxt(filename ,delimiter=";");  #labeled=True;
    #filename+='8cluster2d.txt'; X=np.genfromtxt(filename);  #labeled=True;


    #Generated data
    # X,y=datasets.make_blobs(n_samples=10, random_state=randint(0,50), centers=3, cluster_std=0.8)
    # X,y=datasets.make_blobs(n_samples=n, n_features=2, random_state=42, centers=3, cluster_std=0.8)
    # print(X)
    X, y = datasets.make_moons(n_samples=1000, shuffle=True, noise=0.15, random_state=0)
    #X,labels= SIKCHELP.twomoons_dataset_writer(n, shuffle=0.5, noise=0.05, randoms=randint(0,50), filename= "twomoonstest.csv")

    #X=list(zip(x,y))

    #X,y=datasets.make_circles(n_samples=n, factor=0.7,noise=0.03)

    #Real data
    #filename+='Iris.txt'; X=np.genfromtxt(filename, delimiter=','); labeled=True; X=X[:,1:]


    # #Exoplanet data
    # filename+='expolanet/oec.csv'
    # filename='C:/Users/beer/Desktop/workInProgress/GoKMeansYourself/knnkmeans/examples/exoplanet/oec.csv'
    # df_raw = pd.read_csv(filename)
    # # get rid of nan
    # df = df_raw[df_raw['PlanetIdentifier'].notnull() & df_raw['PlanetaryMassJpt'].notnull() & df_raw['SemiMajorAxisAU'].notnull()]
    # # log of mass and distance to star
    # logM = np.log10(df['PlanetaryMassJpt'])
    # logD = np.log10(df['SemiMajorAxisAU'])
    # # prepare input matrix
    # # write
    # input=[df['PlanetIdentifier'],logM, logD]
    # #to have also a column with the exoplanet names
    # #input = [logM, logD]
    # X = np.matrix(input).T
    # pointsHaveNames=True






    #save labels of points if available
    if labeled:
        labels = X[:, -1]
        X=X[:, :-1]
    if pointsHaveNames:
        names= X[:,0]
        X=X[:,1:]

    #data preparation
    #für iris
    #X=X[:,0:2]
    #X=X[:,1:3]
    #X=X[:,2:4]

    ########TODO add this for real data
    # X=list(zip(X[:,1], X[:,2])) ;
    dim =2

    #X=list(zip(X[:,0], X[:,2]))
    #X=list(zip(X[:,0], X[:,3]))
    #X = list(zip(X[:, 1], X[:, 3]))
    #X=X[:,2:4]
    #X=list(zip(X[:,2], X[:, 4]))
    #X=X[:,:2]
    #X = X[0:3]
    #print('coor: ',X)
    #dim=X.shape[1]

    print("dim", dim)
    #normalize dataset
    #print(f'Data: {X}')
    X = StandardScaler().fit_transform(X)
    #print(f'Data: {X[:20]}')
    #print(f'labels: {y}')

    #represent points as points with positionlist (and labels)
    i=0
    list= None
    pointset=[]
    label=0
    name=""
    for p in X:
        #list=[(p[0],p[1])]
        list=[p]
        if labeled:
            #pointset.append(point.Point(i,list,0,labels[i]))
            #print(labels[i])
            label=labels[i]
        if pointsHaveNames:
            #name=str(names[i][0][0]).replace(r"\(.*\)","")
            name=str(names[i])[3:-3]
            #print("here are the names")
            #print(name)
        pointset.append(Point(i, list, 0, label, name))
        i+=1

    #shuffle all points
    #shuffle(pointset)

    #functions how k changes over time
    n=len(pointset)
    #increase by 1% per step
    def f1(t):
        perc = len(pointset) / 100
        return (t+1)*perc

    #logarithmic
    a = tmax**(2/len(pointset))
    def f2(t):
        return max(math.log(t+2, a), 3)


    #n-th root function
    a= math.log(tmax, 0.5*len(pointset))
    def f3(t):
        return ((t+1)**(1/a) +3)

    #a*sqrt- function
    a= n/ (2*math.sqrt(tmax))
    def f4(t):
        return a * math.sqrt(t+1)

    #linear
    a=(0.5*n-3)/tmax
    def f5(t):
        return a*t + 3

    #linear mit kmax
    def f6(t):
        return t+3

    #linear independent of size of dataset
    def f7(t):
        return 3 + 6*t

    #fibonacci
    def f8(t):
        t+=3
        return ((1 + math.sqrt(5)) **t - (1 - math.sqrt(5)) ** t) / (2 ** t * math.sqrt(5))
    #set t < tmax to get state of the algorithm at time t
    changed=True
    t=0
    imax=25
    #maximal number of desired clusters
    #numberOfDesiredClusters=5
    #tmax= n/numberOfDesiredClusters
    #tmax=(n-3*numberOfDesiredClusters)/(5*numberOfDesiredClusters)
    #print("tmax " ,tmax)
    tmax=15
    #kmax=n/numberOfDesiredClusters
    kmax=math.inf

    def getEpsilon():
        sumdist=0
        counter=0
        #distmin = np.infty
        # for p in X:
        #     for q in X:
        #         dist=distance.euclidean(p,q)
        #         if dist<distmin and dist > 0:
        #             distmin=dist
        #epsilon=distmin
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(X)
        dist, idx = nbrs.kneighbors(X)
        print(nbrs, dist, idx)
        firstNN,secondNN=zip(*dist)
        epsilon=sum(secondNN)/len(secondNN)
        print("epsilon" , epsilon)
        return epsilon/2

    epsilon=getEpsilon()


    while changed and t<tmax and k<=kmax:
        print("t= ", t)
        k=f5(t)
        changed = False
        for point in pointset:
            oldPos=point.getCurrPos()
            # point wanders to new position. if different (>epsilon) from old position, set flag.
            # war für die Experimente vom Paper folgendes:
            # point.setNewKNNPosT(k, pointset, t-1)
            point.setNewKNNPosT(k, pointset, t)
            if distance.euclidean(point.getCurrPos(), oldPos)>epsilon:
                changed=True
        t+=1
    print("t= " ,t)
    #Show plot of contracted points
    
    # plt.figure(1)
    # for p in pointset:
    #     if(dim==4):
    #         x,y,z,w=zip(*p.getPoslis())
    #     if(dim==2):
    #         x,y=zip(*p.getPoslis())
    #     plt.plot(x,y, color="r", zorder=-5)
    # for p in pointset:
    #     pos= p.getCurrPos()
    #     plt.scatter(pos[0], pos[1], color= 'b')

    # title= filename, ' t= ', t, 'compressed data'
    # plt.title(title)
    #plt.axis([-2, 2, -2, 2])

    #plt.show()

    #generate color map to color points according to their clusters
    def get_cmap(N):
        '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
        RGB color.'''
        color_norm  = colors.Normalize(vmin=0, vmax=N-1)
        scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
        def map_index_to_rgb_color(index):
            return scalar_map.to_rgba(index)
        return map_index_to_rgb_color


    #points which now have the same (+-epsilon) coordinate belong to the same cluster
    listOfClusterCenters=[]
    clusterLengths=[0]
    for p in pointset:
        pos=p.getCurrPos()
        i=0
        belongsToACluster=False
        for clusterCenter in listOfClusterCenters:
            #print(clusterCenter)
            if distance.euclidean(pos, clusterCenter)<= epsilon:
                p.setCluster(i)
                belongsToACluster=True
                clusterLengths[i]+=1
                n=clusterLengths[i]
                if dim==4:
                    centerPosX, centerPosY, centerPosZ, centerPosW= (listOfClusterCenters[i])
                    #passe Ort des Clustercenters an
                    listOfClusterCenters[i]= ((centerPosX*(n-1)+pos[0])/n  , (centerPosY*(n-1)+ pos[1])/n, (centerPosZ*(n-1)+ pos[2])/n, (centerPosW*(n-1)+ pos[3])/n)
                if dim==2:
                    #print(listOfClusterCenters[i])
                    centerPosX, centerPosY=(listOfClusterCenters[i])
                    # passe Ort des Clustercenters an
                    listOfClusterCenters[i] = ((centerPosX * (n - 1) + pos[0]) / n, (centerPosY * (n - 1) + pos[1]) / n)

                break
            i+=1
        if not belongsToACluster:
            print("new cluster")
            listOfClusterCenters.append(pos)
            p.setCluster(i)
            clusterLengths.append(1)
    numOfClusters= len(listOfClusterCenters)
    print(numOfClusters, " clusters found")

    from rock import ROCK

    #normalized mutual information
    print("1. ami= ", adjusted_mutual_info_score(y, [p.getCluster() for p in pointset]))

    print("2. ami= ", adjusted_mutual_info_score(y, [p.getLabel() for p in pointset]))
    print("3. ami= ", adjusted_mutual_info_score(y, y))
    print("4. ami= ", adjusted_mutual_info_score(y, ROCK(tmax=15).fit(X).labels_))

    print("old", list([p.getCluster() for p in pointset]))
    print("new", ROCK(tmax=15).fit(X).labels_)
    print("true", y)

    if labeled:
        labels_true=[p.getCluster() for p in pointset]
        labels_pred=[p.getLabel() for p in pointset]
        #print("nmi= ", normalized_mutual_info_score(labels_true, labels_pred))
        print("ari= ", adjusted_rand_score(labels_true,labels_pred))
        print("size of dataset " , len(pointset))

    #show plot of original data colored according to their predicted clusters
    #cmap = get_cmap(numOfClusters+1)
    #plt.figure(2)
    #for p in pointset:
    #    pos= p.getPosT(0)
    #    plt.scatter(pos[0], pos[1], color= cmap(p.getCluster()))
    #    if pointsHaveNames:
    #        plt.annotate(p.getName(), (pos[0]+0.01,pos[1]+0.01), annotation_clip=True)
    #title= filename, ' t= ', t, 'original data'
    #plt.title(title)
    #plt.axis([-2, 2, -2, 2])
    plt.show()
