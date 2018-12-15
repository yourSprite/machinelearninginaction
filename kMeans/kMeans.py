import numpy as np
from matplotlib import pyplot as plt

def loadDataSet(fileName):
    '''
    将文本文件导入列表
    :param fileName:
    :return:
    '''
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curline = line.strip().split('\t')
        fltLine = map(float, curline)
        dataMat.append(list(fltLine))
    return dataMat

def distEclud(vecA, vecB):
    '''
    计算两个向量的欧氏距离
    :param vecA:
    :param vecB:
    :return:
    '''
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

def randCent(dataSet, k):
    '''
    构建随机质心
    :param dataSet:
    :param k:
    :return:
    '''
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2))) # 第一列存放簇类index，第二列存放误差值
    centroids = createCent(dataSet, k) # 初始化质心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m): # 遍历每一行数据，将数据划分至最近的质心
            minDist = np.inf
            minIndex = -1
            for j in range(k): # 寻找最近的质心
                distJI = distMeas(centroids[j, :], dataSet[i, :]) # 计算第i个数据到每个质心的距离
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex: clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
            print(centroids)
        for cent in range(k): # 更新质心位置
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A==cent)[0]] # 获取某个簇类的所有点
            centroids[cent, :] = np.mean(ptsInClust, axis=0) # 计算均值作为新的簇类中心
    return centroids, clusterAssment
