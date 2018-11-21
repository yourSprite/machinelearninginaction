import numpy as np

def loadDataSet():
    '''
    载入数据
    :return:
    '''
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    '''
    sigmoid函数
    :param inX:
    :return:
    '''
    return 1/(1+np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    '''
    梯度上升算法
    :param dataMatIn: 训练数据集
    :param classLabels: 标签
    :return:
    '''
    # 转换为numpy数据类型
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()# 进行转置，行向量转换为列向量
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        # 计算真实类别与计算类别的差值，并按照该差值的方向调整回归系数
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

def plotBestFit(weights):
    '''
    拟合曲线绘图
    :param weights: 最优化方法得到的系数
    '''
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    # 数据散点图
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append((dataArr[i, 2]))
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append((dataArr[i, 2]))
    plt.figure()
    plt.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    plt.scatter(xcord2, ycord2, s=30, c='green')
    x= np.arange(-3.0, 3.0, 0.1)
    # 拟合曲线
    y = (-weights[0]-weights[1]*x) / weights[2]
    plt.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):
    '''
    随机梯度上升
    :param dataMatrix: 特征
    :param classLabels: 标签
    :return: 最优化方法得到的系数
    '''
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    '''
    改进的随机梯度上升算法
    :param dataMatrix:
    :param classLabels:
    :param numIter:
    :return:
    '''
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1+i+j) + 0.1# alpha每次迭代进行调整
            randIndex = int(np.random.uniform(0, len(dataIndex)))# 随机算去更新
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = float(classLabels[randIndex]) - h
            weights = weights + alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def classifyVector(inX, weights):
    '''
    根据sigmoid结果进行分类
    :param inX: 测试向量
    :param weights: 最优化方法得到的系数
    :return: 预测结果
    '''
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:  return 1
    else:   return 0

def colicTest():
    '''
    计算错误率
    :return: 错误率
    '''
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; traingLabels = [];
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        traingLabels.append(currLine[21])
    trainWeights = stocGradAscent1(np.array(trainingSet), traingLabels, 500)
    errorCount = 0; numTestVec = 0
    for line in frTest.readlines():
        numTestVec += 1
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = errorCount / numTestVec
    print('the error rate of this test is:%f' % errorRate)
    return errorRate

def multiTest():
    # 计算平均错误率
    numTests = 10; errorSum = 0
    for k in range(numTests):
        errorSum += colicTest()
    print('after %d iterations the average error rate is:%f' % (numTests, errorSum/float(numTests)))

multiTest()