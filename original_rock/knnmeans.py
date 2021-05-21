from sklearn.neighbors import NearestNeighbors
import numpy as np

'''
point object consists of:
- an ID and 
- a list of positions [e.g.(x,y)] where the last position represents the most current position
- a cluster
'''

class Point:


    def __init__(self, oid, poslis, cluster, label=0, name=""):
        # shall be unique
        self.oid = oid
        # is initialized by one coordinate within list, representing initial position of list
        self.poslis = poslis
        self.cluster = cluster
        self.label = label
        self.name=name


    #returns Point object id
    def getOID(self):
        return self.oid
    
    #returns full list of positions that the Point object ever took
    def getPoslis(self):
        return self.poslis

    #returns last and thus current position of the Point object
    def getCurrPos(self):
        return self.poslis[-1]

    def getPosT(self, t):
        return self.poslis[t]

    #returns the cluster number of the Point object
    def getCluster(self):
        return self.cluster

    #sets the cluster number of the Point object
    def setCluster(self, cluster):
        self.cluster=cluster

    def getLabel(self):
        return self.label

    def setLabel(self, label):
        self.label=label

    def getName(self):
        return self.name

    def setName(self, name):
        self.name=name

    #sets the 'current' position by appending a given coordinate to the end of poslis
    def setCurrPos(self, pos):
        self.poslis.append(pos)
    
    
    #returns the kNN at time t of the Point object, given a list of Point objects
    def getKNNT(self, k, pointlis, t):
        #add self to list of kNN

        k = int(k)

        #TODO which of the next both lines is better?
        ptlis = [self]+pointlis
        #ptlis = pointlis

        #extract a list of coordinates from every point object in pointlis
        coorlis = []
        for e in ptlis:
            # coorlis.append(e.getCurrPos())
            coorlis.append(e.getPosT(t))

        #kNN computation is performed below automagically...
        #in k+1 the '+1' represents the Point object itself
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(coorlis)

        '''
        FUCKING NUMPY MIMIMIMIMI ....go fuck yourself numpy, with your assenite 
        apeshit 'improvements' ! I wish you some thousand fleas crawl up your
        developers candy-ass while paralyzed.
        '''
        X = np.asarray(coorlis[0]).reshape(1,-1)
        nbrskneighbors = nbrs.kneighbors(X)
        dist, idx = nbrskneighbors
        
        #now remap the f****** sklearn indices to our datapoint OIDs:
        knnres = []
        for e,j in zip(idx[0],dist[0]):
            # knnres.append((ptlis[e].getOID(),ptlis[e].getCurrPos(),j))
            knnres.append((ptlis[e].getOID(),ptlis[e].getPosT(t),j))

        return knnres[1:] #slice to exclude the point object itself


    #returns the kNN of the Point object, given a list of Point objects
    def getKNN(self, k, pointlis):
        knns=self.getKNNT(k, pointlis, -1)
        return knns

    
    #takes the kNN and returns the new position of the Point object which is the average 
    #of its kNN
    def setNewKNNPos(self, k, pointlis):
        self.setNewKNNPosT(k, pointlis, -1)

    def setNewKNNPosT(self, k, pointlis, t):
        knnres = self.getKNNT(k, pointlis, t)
        #unzip list of KNN tuples into separate lists --> getting back a list of coordinates
        oidlis, coorlis, distlis = zip(*knnres)
        
        #compute mean of kNN coordinates
        coorarr = np.asarray(coorlis)
        length = coorarr.shape[0]
        dim = (coorarr[0]).shape[0]
        
        #list contains the sum of the values per dimension
        sum_dim_lis = []
        for i in range(dim):
            sum_i = np.sum(coorarr[:, i])
            sum_dim_lis.append(sum_i)
        
        # print('sum_dim_lis: ', sum_dim_lis)
        
        newcoor = list(map(lambda x: x/length, sum_dim_lis))
        self.setCurrPos(newcoor)

    #t: watch positions of other points at time t
    #p: only wander a certain percentage of the way to the kNN center
    def setNewKNNPosTP(self, k, pointlis, t, p):
        oldcoor=list(self.getCurrPos())
        knnres = self.getKNNT(k, pointlis, t)
        # unzip list of KNN tuples into separate lists --> getting back a list of coordinates
        oidlis, coorlis, distlis = zip(*knnres)

        # compute mean of kNN coordinates
        coorarr = np.asarray(coorlis)
        length = coorarr.shape[0]
        dim = (coorarr[0]).shape[0]

        # list contains the sum of the values per dimension
        sum_dim_lis = []
        for i in range(dim):
            sum_i = np.sum(coorarr[:, i])
            sum_dim_lis.append(sum_i)

        newcoor = list(map(lambda x: x / length, sum_dim_lis))
        #vec= newcoor-oldcoor
        # newcoor=tuple(newcoor)
        # oldcoor=tuple(oldcoor)
        # vec= newcoor-oldcoor
        newpos=[None]*len(newcoor)
        print(newcoor , "- newcoor")
        print(oldcoor, "- oldcoor")
        for i in range(len(newcoor)):
            newpos[i]=oldcoor[i]+p*(newcoor[i]-oldcoor[i])
        #newpos=oldcoor + p*(newcoor-oldcoor)
        self.setCurrPos(newpos)
        
    
#teslist of point objects:
# testlis0 = [Point(0, [(2,2)]), Point(1, [(2,5)]), Point(2, [(3,3)]), Point(3, [(1,4)])]
#
# p0 = Point(-1, [(2,3)])
#
#
# p0.getKNN(4,testlis0)
# #    print(e.getOID(), e.getCurrPos())
#
#
# p0.setNewKNNPos(4,testlis0)
# p0.getCurrPos()