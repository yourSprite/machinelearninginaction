import numpy as np

def loadDataSet(filename):
    '''
    加载数据文件
    :param filename: 文件名
    :return: 特征，标签
    '''
    numFeat = len(open(filename).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegres(xArr, yArr):
    '''
    计算最佳拟合直线
    :param xArr: 特征
    :param yArr: 标签
    :return: 回归系数
    '''
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    # 计算行列式，判断xTx是否可逆
    if np.linalg.det(xTx) == 0:
        print('This matrix is singular,cannot do inverse')
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

def lwlr(testPoint, xArr, yArr, k=1.0):
    '''
    局部加权线性回归函数
    :param testPoint: 测试数据
    :param xArr: 训练数据集特征
    :param yArr: 训练数据集标签
    :param k: 参数
    :return: 回归系数
    '''
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))# 初始化权重
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))# 权重值大小以指数级衰减
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print('This matrix is singular,cannot do inverse')
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    '''
    测试lwlr
    :param testArr: 测试数据集
    :param xArr: 测试数据集特征
    :param yArr: 测试数据集标签
    :param k: 核参数
    :return: 预测值
    '''
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def rssError(yArr, yHatArr):
    '''
    分析预测误差大小
    :param yArr: 标签值
    :param yHatArr: 预测值
    :return: 错误率
    '''
    return ((yArr-yHatArr)**2).sum()

def ridgeRegress(xMat, yMat, lam=0.2):
    '''
    岭回归
    :param xMat: 测试数据集特征
    :param yMat: 测试数据集标签
    :param lam: 参数lambda
    :return: 回归系数
    '''
    xTx = xMat.T * xMat
    demon = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(demon) == 0.0:
        print('This matrix is singular,cannot do inverse')
        return
    ws = demon.I * (xMat.T*yMat)
    return ws

def ridgeTest(xArr, yArr):
    '''
    测试岭回归
    :param xArr: 测试数据集特征
    :param yArr: 测试数据集标签
    :return: 系数
    '''
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    # 数据标准化，所有特征减去各自的均值并处以方差
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat-xMeans)/xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegress(xMat, yMat, np.exp(i-10))
        wMat[i, :] = ws.T
    return wMat

def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = np.mean(inMat,0)   #calc mean then subtract it off
    inVar = np.var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

def stageWise(xArr, yArr, eps=0.01, numIt=100):
    '''
    逐步线性回归
    :param xArr: 数据
    :param yArr: 预测变量
    :param eps: 每次迭代需要调整的步长
    :param numIt: 迭代次数
    :return: 回归系数
    '''
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = np.inf# 最小误差初始化为正无穷
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat