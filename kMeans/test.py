import kMeans
import numpy as np

# dataMat = np.mat(kMeans.loadDataSet('testSet.txt'))
# print(kMeans.randCent(dataMat, 2))
# print(kMeans.distEclud(dataMat[0], dataMat[1]))

dataMat = np.mat(kMeans.loadDataSet('testSet.txt'))
myCentroids, clustAssing = kMeans.kMeans(dataMat, 4)