import numpy as np

def loadSimData():
    datMat = np.matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    '''
    通过阈值比较对数据进行分类
    :param dataMatrix: 输入数据集
    :param dimen: 特征索引
    :param threshVal: 阈值
    :param threshIneq: 阈值类别
    :return: 分类特征
    '''
    retArry = np.ones((np.shape(dataMatrix)[0], 1))# 将返回数组的全部元素设置为1
    # 小于阈值数据分到-1，另一边为+1
    if threshIneq == 'lt':
        retArry[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArry[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArry

def buildStump(dataArr, classLabels, D):
    '''
    单层决策树生成函数
    :param dataArr: 输入数据集
    :param classLabels: 标签
    :param D: 权重向量
    :return: 最小错误率单层决策树，最小错误率，估计的类别向量
    '''
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}# 用于储存给定向量D时所得到的最佳单层决策树相关信息
    minError = np.inf
    # 遍历所有特征
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin)/numSteps# 通过最大值最小值来获取步长
        # 遍历这些值
        for j in range(-1, int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr [predictedVals == labelMat] = 0
                weightedError = D.T*errArr
                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                # 当前的错误率小，就在词典bestStump中保存该单层决策树
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    '''
    基于单层决策树的Adaboost训练过程
    :param dataArr:
    :param classLabels:
    :param numIt: 迭代次数
    :return:
    '''
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1))/m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        # print("D:", D.T)
        alpha = float(0.5*np.log((1.0-error)/max(error, 1e-16)))# 确保不会发生除零溢出
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        # print("classEst:", classEst.T)
        # 为下一次迭代计算D
        expon = np.multiply(-1*alpha*np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()
        # 错误率累加计算
        aggClassEst += alpha*classEst
        # print("aggClassEst:", aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum()/m
        # print("total error:", errorRate, "\n")
        if errorRate == 0.0: break
    return weakClassArr

def adaClassify(dataToClass, classifierArr):
    '''
    AdaBoost分类函数
    :param dataToClass: 待分类样例
    :param classifierArr: 多个弱分类器组成的数组
    :return:
    '''
    dataMatrix = np.mat(dataToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    # 遍历classifierArr中所有弱分类器
    for i in range(len(classifierArr)):
        # 基于stumpClassify()对每个分类器得到一个类别的估计值
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return np.sign(aggClassEst)

def plotROC(predStrengths, classLabels):
    '''
    ROC曲线的绘制及AUC计算函数
    :param predStrengths: 分类器的预测强度
    :param classLabels:
    '''
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)# 绘制光标的位置
    ySum = 0.0# AUC的值
    numPosClas = sum(np.array(classLabels)==1.0)
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = 0
        else:
            delX = xStep
            delY = 0
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Postive Rate')
    plt.ylabel('True Positive rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print('the Area Under the Curve is:', ySum*xStep)