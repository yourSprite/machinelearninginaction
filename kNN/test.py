import numpy as np
import operator
import kNN
import matplotlib
import matplotlib.pyplot as plt
import os

# def classify0(inX, dataSet, labels, k):
# inX = [0, 0]
# k = 3
# labels = ['一', '二', '二']
# dataSet = np.array([[2, 2], [3, 3], [1, 1]])
# dataSize = dataSet.shape[0]
# diffMat = np.tile(inX, (dataSize, 1)) - dataSet
# sqDiffMat = diffMat ** 2
# sqDistances = sqDiffMat.sum(axis=1)
# distances = sqDistances ** 0.5
# sortedDistindices = distances.argsort()
# classCount = {}
# for i in range(3):
#     voteIlabel = labels[sortedDistindices[i]]
#     classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
# sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
# print(sortedClassCount[0][0])



# dict = {'ceshi1':1, 'ceshi2':2}
# print(dict.iteritems())
#
# filename = 'datingTestSet.txt'
# fr = open(filename)
# arrayOLines = fr.readlines()
# numberOfLines = len(arrayOLines)
# returnMat = np.zeros((numberOfLines, 3))
# classLabel = []
# index = 0
# for line in arrayOLines:
#     line.strip()
#     listFromLine = line.split('\t')
#     returnMat[index, :] = listFromLine[0 : 3]
#     classLabel.append(listFromLine[-1].strip())
#     index += 1
# print(classLabel)


# datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
# plt.figure()
# plt.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
# color = np.random.randint(0,10,2)# 颜色
# plt.scatter(datingDataMat[:, 0], datingDataMat[:, 1], c=['r', 'b'])
# plt.show()

# datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
# dataSet = datingDataMat
# minVals = dataSet.min(0)
# maxVals = dataSet.max(0)
# ranges = maxVals - minVals
# m = dataSet.shape[0]
# normanDataSet = (dataSet - np.tile(minVals, (m, 1))) / (np.tile(maxVals, (m, 1)) - np.tile(minVals, (m, 1)))
# normanDataSet, ranges, minVals = kNN.autoNorm(datingDataMat)
# print(normanDataSet)

# hoRatio = 0.1
# datingDataMat, datingLabels = kNN.file2matrix('datingTestSet.txt')
# normanDataSet, ranges, minVals = kNN.autoNorm(datingDataMat)
# m =  normanDataSet.shape[0]
# numTestVecs = int(m*hoRatio)
# errorCount = 0
# for i in range(numTestVecs):
#     classifierResult = kNN.classify0(normanDataSet[i, :], normanDataSet[numTestVecs:, :], datingLabels[numTestVecs:], 3)
#     # print(classifierResult)
#     if (classifierResult != datingLabels[i]) : errorCount += 1
#
# print("错误率为：%f" % (errorCount/(m*hoRatio)))

# persentTags = input('测试')
# print(persentTags)

# 从文件名解析分类数字（训练集），将特征载入矩阵，标签存入列表
trainingFileList = os.listdir('trainingDigits')
m = len(trainingFileList)
trainingMat = np.zeros((m, 1024))
hwLabels = []
for i in range(m):
    classNumStr = int(trainingFileList[i].split('.')[0].split('_')[0])
    trainingMat[i, :] = kNN.img2vector('trainingDigits/%s' % trainingFileList[i])
    hwLabels.append(classNumStr)
# 从文件名解析分类数字（测试集）
testFileList = os.listdir('testDigits')
mTest = len(testFileList)
testMat = np.zeros((m, 1024))
errorCount = 0
for i in range(mTest):
    classNumStr = int(testFileList[i].split('.')[0].split('_')[0])
    vectorUnderTest = kNN.img2vector('testDigits/%s' % testFileList[i])
    classifierResult = kNN.classify0(vectorUnderTest, trainingMat, hwLabels, 3)
    if (classNumStr != classifierResult): errorCount += 1
print('错误率为：%f' % (errorCount/mTest))