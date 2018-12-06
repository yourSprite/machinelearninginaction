import numpy as np


def loadDataSet(fileName):
    '''
    加载数据集
    :param fileName:
    :return:
    '''
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)  # 把每行映射成浮点数
        dataMat.append(list(fltLine))
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    '''
    分隔数据集
    :param dataSet: 数据集
    :param feature: 待切分的特征
    :param value: 切分值
    :return:
    '''
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):
    '''
    生成叶节点（目标变量的均值）
    :param dataSet:
    :return:
    '''
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    '''
    在给定数据上计算目标变量的平方误差（总方差   ）
    :param dataSet:
    :return:
    '''
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    '''
    回归树的切分函数，找到数据的最佳二元切分发
    :param dataSet:
    :param leafType:
    :param errType:
    :param ops:
    :return:
    '''
    tolS = ops[0]  # 容许的误差下降值
    tolN = ops[1]  # 切分的最小样本数
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:  # 如果所有制相等则退出
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n - 1):
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN): continue  # 某个子集小于tolN，不切分
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:  # 如果误差减小不大则退出
        return None, leafType(dataSet)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):  # 如果切分出的数据集很小则退出
        return None, leafType(dataSet)
    return bestIndex, bestValue


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    '''
    CART算法创建回归数
    :param dataSet:
    :param leafType: 建立叶节点的函数
    :param errType: 误差计算函数
    :param ops: 包含树构建所需其他参数的元组
    :return: 树字典
    '''
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None: return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


def isTree(obj):
    '''
    判断当前节点是子树还是节点
    :param obj:
    :return: 子树返回true，节点返回false
    '''
    return (type(obj).__name__ == 'dict')


def getMean(tree):
    '''
    从上往下遍历直到叶节点
    :param tree: 树
    :return: 树平均值（塌陷处理）
    '''
    if isTree(tree['right']): tree['right'] = getMean(tree['reght'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, testData):
    '''
    回归树剪枝函数
    :param tree: 树
    :param testData: 测试数据
    :return: 剪枝之后的树
    '''
    if np.shape(testData)[0] == 0: return getMean(tree)  # 没有测试数据则对树进行塌陷处理
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = np.sum(np.power(lSet[:, -1] - tree['left'], 2)) + \
                       np.sum(np.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = np.sum(np.power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print('merging')
            return treeMean
        else:
            return tree
    else:
        return tree


def linearSolve(dataSet):
    '''
    将数据集格式化成目标变量Y和自变量X，并进行线性回归
    :param dataSet:
    :return:
    '''
    m, n = np.shape(dataSet)
    X = np.mat(np.ones((m, n)))
    Y = np.mat(np.ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n - 1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0:  # 求行列式判断矩阵是否可逆
        raise NameError('This matrix is singular, cannot do inverse, \n\
                        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)  # 简单的线性回归
    return ws, X, Y


def modelLeaf(dataSet):
    '''
    返回回归系数
    :param dataSet:
    :return:
    '''
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    '''
    计算平方误差
    :param dataSet:
    :return:
    '''
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return np.sum(np.power(Y - yHat, 2))


def regTreeEval(model, inDat):
    return float(model)


def modelTreeEval(model, inDat):
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)


def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat
