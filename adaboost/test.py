import adaboost
import numpy as np

# dataMat, classLabels = adaboost.loadSimData()
# D = np.mat(np.ones((5, 1))/5)
# adaboost.buildStump(dataMat, classLabels, D)
# classifierArray = adaboost.adaBoostTrainDS(dataMat, classLabels, 9)
# a = np.sign(-50)
# print(a)

# dataArr, labelArr = adaboost.loadSimData()
# classifierArr = adaboost.adaBoostTrainDS(dataArr, labelArr)
# adaboost.adaClassify([0, 0], classifierArr)
# adaboost.adaClassify([[5, 5], [0, 0]], classifierArr)

# dataArr, labelArr = adaboost.loadDataSet('horseColicTraining2.txt')
# classifierArray, aggClassEst = adaboost.adaBoostTrainDS(dataArr, labelArr,10)
# adaboost.plotROC(aggClassEst.T, labelArr)