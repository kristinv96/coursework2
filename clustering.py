import numpy as np
import random
import matplotlib.pyplot as plt

np.warnings.filterwarnings('ignore')

iterations=input("Enter number of iterations for the assignment and optimization phase:")
iterations=int(iterations)
random.seed(2)#This is to have all the same random points to compare different answers to the questions

#The stage that is performed first.
def initialPhase(k):
    #Read the data from each file. Create a function that does this?
    #We want to end up with a numpy array (matrix) that contains all the data. 
    #So a whole array with all the points, from animals, countries, fruits and veggies.
    #Also we dont want to include the first item of the rows, which is a name of type string.
    dataFilenames = ["animals","countries","fruits","veggies"]

    dataArray = np.empty((0,300), float)
    datalen = []
    trueLabels = [] #Initializes a list that keeps track of the true labels of the data
    for i in dataFilenames:
        data1 = np.genfromtxt(i, delimiter=' ', usecols=range(1,301),dtype=float) #creates a numpy.ndarray object
        datalen.append(len(data1))
        #Put the data from the file to the large array with all the data
        dataArray = np.append(dataArray,data1,axis= 0)
        for j in data1:
            if i=="animals":
                trueLabels.append(1)
            elif i=="countries":
                trueLabels.append(2)
            elif i=="fruits":
                trueLabels.append(3)
            elif i=="veggies":
                trueLabels.append(4)

    dRows = dataArray.shape[0] #prints out 327
    dCols = dataArray.shape[1] #prints out 300

    #Now we have our ndarray(matrix) that contains all the points from all the datasets. That is called dataArray.
    #We want to pick k (here k=4) random points in the whole data. So 4 different points in dataArray
    randomPoints = np.empty((k,dCols), float) #Initializes an array we will use to keep track of the random points. Same as saying (4,300)

    i=0
    for _ in range(k): #For-loop that generates the random points
        #random row
        a = random.randint(0,dRows) #both numbers are included
        if a in randomPoints:
            while(a in randomPoints):
                a = random.randint(0,dRows)
        #print(a)
        b = dataArray[a]
        randomPoints[i]=b #Let's put the random point into the randomPoints array that keeps track of it
        i+=1
    return randomPoints,dataArray,trueLabels
#The optimization+assignment phase, using means
def AOmeansBcubed(centroidPoints,iters,dataArray,k,nor):
    newMeans = np.empty((0,300), float) #empty numpy array with dimensions 4 and 300
    r = False
    while(iters > 0):#runs as many times as the variable iters says it to
        #Assignment phase
        #Each cluster will be a ndarray of points. We can keep track of it by just 
        #referring to the index of the point in the dataArray (since it doesn't change through our program). Let's keep the indexes as items in a list,
        #and we have k lists. We keep the k lists in a dict.
        if r == True:
            centroidPoints = newMeans
        clusters = [] #Initializes a list that keeps track of the clusters
        indexes = []
        #for every point in the dataset, calculate the squared euclidian distance between the randomPoint=centroid
        count=0
        for j in dataArray:
            indexes.append(count)
            minDist = float("inf") #define the minimum distance as a really big number initially
            distDict = {} #initialize the distance dict for each point j
            for i in range(0,k): #do this k times, because we want to find the minimum distance
                centroid=centroidPoints[i]
                if nor==True:#normalize both the point j and the centoid, if true
                    j= j / np.linalg.norm(j)
                    centroid=centroid/np.linalg.norm(centroid)
                euclidianDist = np.linalg.norm(j-centroid) #distance between j and the centroid
                squaredED = np.power(euclidianDist,2) #squared
                distDict[(i+1)] = squaredED #Add the squared euclidian distance into the dict. The key refers to the centroid that we are currently looking at. They are k many.
                #if distance=0 then continue to the next iteration in the for-loop, that is go to the next point to measure distance. We don't want
                #to add 0 as the minDist
                if(squaredED == 0):
                    continue
                if minDist > squaredED:
                    minDist = squaredED #if this is bad maybe try min(d, key=d.get) in the dict. But we cannot include 0, need to implement that somehow
            #now assign the point to the nearest centroid. We have the minimum distance and can refer to the key in the distDict to assign the point to a cluster.
            if k==1:#in the case of k being 1, we only have one point to assign the cluster to
                c=1
            else:
                c = list(distDict.keys())[list(distDict.values()).index(minDist)] #this line writes out the key for the value minDist in distDict. Basically writing out number of k which the current point j is assigned to
            count += 1 #increment the count of points. So the same as the index of dataArray
            #print(distDict)
            clusters.append(c)
            #the index of the list is the same as the index of the datapoint in dataArray

        #initialize a list that has k(here 4) rows
        clusterLen = []
        #print(clusters)
        #Want to create k many lists. (nparrays)
        for j in range(1,k+1):
            counter = 0
            for i in range(len(clusters)): #goes over all items in clusters array.0,1,2,...,325,326
                if clusters[i] == j:
                    counter +=1
            clusterLen.append(counter)

        l = []
        for i in range(k):
            l.append([])

        for i in range(len(indexes)):
            l[clusters[i]-1].append(indexes[i])

        clusterArr=np.array([np.array(xi) for xi in l],dtype=object)

        #print(clusterArr)

        #Optimization phase

        #Here we are going to use the arrays dataArray (that contains the points) and clusterArr (that contains the indexes of the points in each cluster)
        #to calculate the mean for each group:

        newMeans = np.empty((0,300), float) #empty numpy array with dimensions 4 and 300

        for i in range(k):
            #create a numpy array A containing all datapoints with the corresponding indexes kept in clusterArr. So this is done for k=1, then k=2...Hence
            #the for loop.

            #we are looking at array no. i now.
            le=len(clusterArr[i])#length of this cluster. How many points are in cluster i
            currentData = np.empty((0,300), float) #has le rows, and dCols columns
            #for-loop that goes on le times, and adds the corresponding dataPoint into the new array.
            
            for j in range(0,le):
                
                c=clusterArr[i][j]#current number(index number) we are looking at in the cluster array

                dataArray[c]
                d=dataArray[c]
                #print(d.shape)
                currentData = np.vstack([currentData,d])
            #print(currentData.shape) #this is correct
            
            #print(currentData)
            #calculate mean of all the points in A
            meanCurrent=np.mean(currentData,axis = 0)
            #print("k=",k)
            #print(meanCurrent)
            #store the mean in the numpy array newMeans already initialized. It contains k points, so here 4 points with 300 dimensions.
            newMeans = np.vstack([newMeans,meanCurrent]) 
        #print(newMeans) #this here prints out the new optimized means! So the new randompoints
        #print(newMeans.shape) #prints out (4,300)
        iters-=1
        r = True
    #When we have our clusters array ready:
    #Now we use the clusters list to calculate the b-cubed recall, precision and f-score. The REAL group boundaries are as follows: THERE
    #ARE FOUR REAL GROUPS, THAT NEVER CHANGES EVEN IF K CHANGES:
    group1 = clusters[:50]#animals
    group2 = clusters[50:211]#countries
    group3 = clusters[211:269]#fruits
    group4 = clusters[269:327]#veggies

    groupArr = [group1,group2,group3,group4]

    recallArr = []
    precisionArr = []
    fscoreArr = []

    totalLabels=k
    occurences={}
    totalOccurences={}
    #count occurences of each number over all groups.
    for j in range(1,totalLabels+1):
        o=clusters.count(j)
        totalOccurences[j]=o
    
    #print("Total occurences:",totalOccurences)

    #Loop through all groups in groupArr
    for i in groupArr:
        m=len(i)#total points in group
        #use k to find out how many numbers we are dealing with
        for j in range(1,totalLabels+1):
            occurence=i.count(j)#i:group ; j:label we are want to count. Could be 1, 2,3 for example
            occurences[j]=occurence #now we have filled the dictionary with the occurences. So key is the label, value is its occurence
        #print("Occurences:",occurences)
        
        #Loop through all points in the particular group.
        for j in i:#In a particular point, calculate recall, precision and fscore and store in the earlier intialized arrays.
            currentLabel=j
            n=occurences[currentLabel] #total occurences of label in the particular group
            recall=n/m
            recallArr.append(recall)
            w=totalOccurences[j] #fetch from our dictionary totalOccurences
            precision=n/w
            precisionArr.append(precision)
            fscore=2*((precision*recall)/(precision+recall))
            fscoreArr.append(fscore)
    
    #Now let's calculate our final recall, precision and fscore by using the average number.
    rl=len(recallArr)
    pl=len(precisionArr)
    fl=len(fscoreArr)

    #print(fscoreArr)

    finalRecall=sum(recallArr)/rl
    finalPrecision=sum(precisionArr)/pl
    finalFscore=sum(fscoreArr)/fl

    #print("sum(fscoreArr)",sum(fscoreArr))
    #print("fl",fl)
    #print("rl",rl)
    #print("pl",pl)
    #print(finalRecall)
    #print(finalPrecision)
    #print(finalFscore)

    return newMeans,finalRecall,finalPrecision,finalFscore
#The optimization+assignment phase, using medians
def AOmediansBcubed(centroidPoints,iters,dataArray,k,nor):
    newMedians = np.empty((0,300), float) #empty numpy array with dimensions 4 and 300
    r = False
    while(iters > 0):#runs as many times as the variable iters says it to
        #Assignment phase
        #Each cluster will be a ndarray of points. We can keep track of it by just 
        #referring to the index of the point in the dataArray (since it doesn't change through our program). Let's keep the indexes as items in a list,
        #and we have k lists. We keep the k lists in a dict.
        if r == True:
            centroidPoints = newMedians
        clusters = [] #Initializes a list that keeps track of the clusters
        indexes = []
        #for every point in the dataset, calculate the squared euclidian distance between the randomPoint=centroid
        count=0
        for j in dataArray:
            indexes.append(count)
            minDist = float("inf") #define the minimum distance as a really big number initially
            distDict = {} #initialize the distance dict for each point j
            for i in range(0,k): #do this k times, because we want to find the minimum distance
                centroid=centroidPoints[i]
                if nor==True:#normalize both the point j and the centoid, if true
                    j= j / np.linalg.norm(j)
                    centroid=centroid/np.linalg.norm(centroid)
                #let's calculate the manhattan distance:
                manhattan = np.sum(np.absolute(centroid-j), axis = 0)
                distDict[(i+1)] = manhattan #Add the squared euclidian distance into the dict. The key refers to the centroid that we are currently looking at. They are k many.
                #if distance=0 then continue to the next iteration in the for-loop, that is go to the next point to measure distance. We don't want
                #to add 0 as the minDist
                if(manhattan == 0):
                    continue
                if minDist > manhattan:
                    minDist = manhattan #if this is bad maybe try min(d, key=d.get) in the dict. But we cannot include 0, need to implement that somehow
            #now assign the point to the nearest centroid. We have the minimum distance and can refer to the key in the distDict to assign the point to a cluster.
            if k==1:#in the case of k being 1, we only have one point to assign the cluster to
                c=1
            else:
                c = list(distDict.keys())[list(distDict.values()).index(minDist)] #this line writes out the key for the value minDist in distDict. Basically writing out number of k which the current point j is assigned to
            count += 1 #increment the count of points. So the same as the index of dataArray
            clusters.append(c) #the index of the list is the same as the index of the datapoint in dataArray

        #initialize a list that has k(here 4) rows
        clusterLen = []

        #Want to create k many lists. (nparrays)
        for j in range(1,k+1):
            counter = 0
            for i in range(len(clusters)): #goes over all items in clusters array.0,1,2,...,325,326
                if clusters[i] == j:
                    counter +=1
            clusterLen.append(counter)

        l = []
        for i in range(k):
            l.append([])

        for i in range(len(indexes)):
            l[clusters[i]-1].append(indexes[i])

        clusterArr=np.array([np.array(xi) for xi in l],dtype=object)

        #print(clusterArr)

        #Optimization phase

        #Here we are going to use the arrays dataArray (that contains the points) and clusterArr (that contains the indexes of the points in each cluster)
        #to calculate the mean for each group:

        newMedians = np.empty((0,300), float) #empty numpy array with dimensions 4 and 300

        for i in range(k):
            #create a numpy array A containing all datapoints with the corresponding indexes kept in clusterArr. So this is done for k=1, then k=2...Hence
            #the for loop.

            #we are looking at array no. i now.
            le=len(clusterArr[i])#length of this cluster. How many points are in cluster i
            currentData = np.empty((0,300), float) #has le rows, and dCols columns
            #for-loop that goes on le times, and adds the corresponding dataPoint into the new array.
            
            for j in range(0,le):
                
                c=clusterArr[i][j]#current number(index number) we are looking at in the cluster array

                dataArray[c]
                d=dataArray[c]
                #print(d.shape)
                currentData = np.vstack([currentData,d])
            #print(currentData.shape) #this is correct
            
            #print(currentData)
            #calculate mean of all the points in A
            medianCurrent=np.median(currentData,axis = 0)
            #print("k=",k)
            #print(meanCurrent)
            #store the mean in the numpy array newMeans already initialized. It contains k points, so here 4 points with 300 dimensions.
            newMedians = np.vstack([newMedians,medianCurrent]) 

        #print(newMeans) #this here prints out the new optimized means! So the new randompoints
        #print(newMeans.shape) #prints out (4,300)
        iters-=1
        r = True
    #When we have our clusters array ready:
    #Now we use the clusters list to calculate the b-cubed recall, precision and f-score. The REAL group boundaries are as follows: THERE
    #ARE FOUR REAL GROUPS, THAT NEVER CHANGES EVEN IF K CHANGES:
    group1 = clusters[:50]#animals
    group2 = clusters[50:211]#countries
    group3 = clusters[211:269]#fruits
    group4 = clusters[269:327]#veggies

    groupArr = [group1,group2,group3,group4]

    recallArr = []
    precisionArr = []
    fscoreArr = []

    totalLabels=k
    occurences={}
    totalOccurences={}
    #count occurences of each number over all groups.
    for j in range(1,totalLabels+1):
        o=clusters.count(j)
        totalOccurences[j]=o
    
    #print(totalOccurences)

    #Loop through all groups in groupArr
    for i in groupArr:
        m=len(i)#total points in group
        #use k to find out how many numbers we are dealing with
        for j in range(1,totalLabels+1):
            occurence=i.count(j)#i:group ; j:label we are want to count. Could be 1, 2,3 for example
            occurences[j]=occurence #now we have filled the dictionary with the occurences. So key is the label, value is its occurence
        #print(occurences)
        
        #Loop through all points in the particular group.
        for j in i:#In a particular point, calculate recall, precision and fscore and store in the earlier intialized arrays.
            currentLabel=j
            n=occurences[currentLabel] #total occurences of label in the particular group
            recall=n/m
            recallArr.append(recall)
            w=totalOccurences[j] #fetch from our dictionary totalOccurences
            precision=n/w
            precisionArr.append(precision)
            fscore=2*((precision*recall)/(precision+recall))
            fscoreArr.append(fscore)
    
    #Now let's calculate our final recall, precision and fscore by using the average number.
    finalRecall=sum(recallArr)/len(recallArr)
    finalPrecision=sum(precisionArr)/len(precisionArr)
    finalFscore=sum(fscoreArr)/len(fscoreArr)

    #print(finalRecall)
    #print(finalPrecision)
    #print(finalFscore)

    return newMedians,finalRecall,finalPrecision,finalFscore

print('-------------------------------------------------------')
print("Question 3:")#K-means unnormalized
print("K-means unnormalized")
normalize=False #Variable that indicates if we want to normalize the points or not
k2=[1,2,3,4,5,6,7,8,9]
p=[]
r=[]
f=[]
for ki in k2:#let's do this for all the k's in the k2 array
    result = initialPhase(ki)
    r11=result[0]#randomPoints
    r12=result[1]#dataArray
    aa=AOmeansBcubed(r11,iterations,r12,ki,normalize)
    a22=aa[1]#finalRecall
    a33=aa[2]#finalPrecision
    a44=aa[3]#finalFscore
    r.append(a22)
    p.append(a33)
    f.append(a44)

xpoints = np.array(k2)
y1 = np.array(r)
y2 = np.array(p)
y3 = np.array(f)

plt.title("B-CUBED Recall, Precision, F-score")
plt.suptitle("K-means unnormalized")
plt.xlabel("k")
plt.ylabel("")

line1, = plt.plot(xpoints, y1)
line2, = plt.plot(xpoints, y2)
line3, = plt.plot(xpoints, y3)

plt.legend(handles = [line1, line2, line3], 
           labels  = ["Recall", "Precision","F-score"])
plt.show()

print('-------------------------------------------------------')
print("Question 4:")#K-means normalized
print("K-means normalized")
normalize=True #Variable that indicates if we want to normalize the points or not
k2=[1,2,3,4,5,6,7,8,9]
p=[]
r=[]
f=[]
for ki in k2:#let's do this for all the k's in the k2 array
    result = initialPhase(ki)
    r11=result[0]#randomPoints
    r12=result[1]#dataArray
    aa=AOmeansBcubed(r11,iterations,r12,ki,normalize)
    a22=aa[1]#finalRecall
    a33=aa[2]#finalPrecision
    a44=aa[3]#finalFscore
    r.append(a22)
    p.append(a33)
    f.append(a44)

xpoints = np.array(k2)
y1 = np.array(r)
y2 = np.array(p)
y3 = np.array(f)

plt.title("B-CUBED Recall, Precision, F-score")
plt.suptitle("K-means normalized")
plt.xlabel("k")
plt.ylabel("")

line1, = plt.plot(xpoints, y1)
line2, = plt.plot(xpoints, y2)
line3, = plt.plot(xpoints, y3)

plt.legend(handles = [line1, line2, line3], 
           labels  = ["Recall", "Precision","F-score"])
plt.show()

print('-------------------------------------------------------')
print("Question 5:")#K-medians unnormalized
print("K-medians unnormalized")
normalize=False #Variable that indicates if we want to normalize the points or not
k2=[1,2,3,4,5,6,7,8,9]
p=[]
r=[]
f=[]
for ki in k2:#let's do this for all the k's in the k2 array
    result = initialPhase(ki)
    r11=result[0]#randomPoints
    r12=result[1]#dataArray
    aa=AOmediansBcubed(r11,iterations,r12,ki,normalize)
    a22=aa[1]#finalRecall
    a33=aa[2]#finalPrecision
    a44=aa[3]#finalFscore
    r.append(a22)
    p.append(a33)
    f.append(a44)

xpoints = np.array(k2)
y1 = np.array(r)
y2 = np.array(p)
y3 = np.array(f)

plt.title("B-CUBED Recall, Precision, F-score")
plt.suptitle("K-medians unnormalized")
plt.xlabel("k")
plt.ylabel("")

line1, = plt.plot(xpoints, y1)
line2, = plt.plot(xpoints, y2)
line3, = plt.plot(xpoints, y3)

plt.legend(handles = [line1, line2, line3], 
           labels  = ["Recall", "Precision","F-score"])
plt.show()

print('-------------------------------------------------------')
print("Question 6:")#K-medians normalized
print("K-medians normalized")
normalize=True #Variable that indicates if we want to normalize the points or not
k2=[1,2,3,4,5,6,7,8,9]
p=[]
r=[]
f=[]
for ki in k2:#let's do this for all the k's in the k2 array
    result = initialPhase(ki)
    r11=result[0]#randomPoints
    r12=result[1]#dataArray
    aa=AOmediansBcubed(r11,iterations,r12,ki,normalize)
    a22=aa[1]#finalRecall
    a33=aa[2]#finalPrecision
    a44=aa[3]#finalFscore
    r.append(a22)
    p.append(a33)
    f.append(a44)

xpoints = np.array(k2)
y1 = np.array(r)
y2 = np.array(p)
y3 = np.array(f)

plt.title("B-CUBED Recall, Precision, F-score")
plt.suptitle("K-medians normalized")
plt.xlabel("k")
plt.ylabel("")

line1, = plt.plot(xpoints, y1)
line2, = plt.plot(xpoints, y2)
line3, = plt.plot(xpoints, y3)

plt.legend(handles = [line1, line2, line3], 
           labels  = ["Recall", "Precision","F-score"])
plt.show()