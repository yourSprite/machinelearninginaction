import regression
import numpy as np
import matplotlib.pyplot as plt

# xArr, yArr = regression.loadDataSet('ex0.txt')
# ws = regression.standRegres(xArr, yArr)
# xMat = np.mat(xArr)
# yMat = np.mat(yArr)
# yHat = xMat * ws
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
# xCopy = xMat.copy()
# # xCopy.sort(0)
# yHat = xCopy * ws
# ax.plot(xCopy[:, 1], yHat)
# plt.show()
# # print(np.corrcoef(yHat.T, yMat))

# xArr, yArr = regression.loadDataSet('ex0.txt')
# xMat = np.mat(xArr)
# # ws1 = regression.lwlr(xArr[0], xArr, yArr, 1.0)# 对单点进行估计
# # ws2 = regression.lwlr(xArr[0], xArr, yArr, 0.001)
# yHat = regression.lwlrTest(xArr, xArr, yArr, 0.003)
# strInd = xMat[:, 1].argsort(0)
# xSort = xMat[strInd][:, 0, :]
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(xSort[:, 1], yHat[strInd])
# ax.scatter(xMat[:, 1].flatten().A[0], np.mat(yArr).T[:, 0].flatten().A[0], s=2, c='red')
# plt.show()

# abX, abY = regression.loadDataSet('abalone.txt')
# ridgeWeights = regression.ridgeTest(abX, abY)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(ridgeWeights)
# plt.show()

xArr, yArr = regression.loadDataSet('abalone.txt')
regression.stageWise(xArr, yArr, 0.01, 200)