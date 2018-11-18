from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def calcShannonEnt(dataSet):
    '''
    计算给定数据集的香农熵
    :param dataSet: 需要计算的数据集
    :return: 香农熵
    '''
    numEntries = len(dataSet)
    labelCounts = {}
    # 取出标签并统计出现次数保存在字典中
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # 计算标签的香农商
    shannonEnt = 0
    for key in labelCounts:
        prob = labelCounts[key] / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    '''
    按照给定特征划分数据集
    :param dataSet: 待划分的数据集
    :param axis: 划分数据集的特征，即根据第axis列的数据进行划分
    :param value: 需要返回的特征值
    :return: 划分好的数据集
    '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    '''
    选择最好的数据集划分方式
    :param dataSet: 数据
    :return: 特征索引
    '''
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0; bestFeature = -1
    # 遍历所有特征
    for i in range(numFeatures):
        # 遍历所有特征可能的值
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0
        # 遍历属性，对每一个属性划分数据集
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算新划分数据集的熵值并进行求和
            prob = len(subDataSet) / len(dataSet)
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    '''
    多数表决方法，统计分类数量，取数量最多的分类
    :param classList: 列表
    :return: 结果
    '''
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted()(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return  sorted()

def createTree(dataSet, labels):
    '''
    创建树
    :param dataSet:数据集
    :param labels: 标签
    :return: 决策树
    '''
    classList = [example[-1] for example in dataSet]
    # 递归停止条件，所有类标签相同
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 递归停止条件，使用完了所有特征
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    # 取得特征所有属性
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    # 在每个属性上递归调用createTree
    for value in uniqueVals:
        subLabels = labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

myDat, labels = createDataSet()
myTree = createTree(myDat, labels)
print(myTree)