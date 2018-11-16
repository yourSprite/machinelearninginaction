import numpy as np
import operator
import os


def createDataSet():
    '''
    数据准备
    '''
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    '''
    kNN算法
    :param inX: 用于分类的输入向量
    :param dataSet: 训练样本
    :param labels: 标签向量
    :param k: 近邻数量
    :return: 分类结果
    '''
    # 距离计算
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistindices = distances.argsort()
    classCount = {}
    # 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistindices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        # 排序
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    '''
    解析文本文件
    :param filename: 文件名
    :return: 特征和标签
    '''
    fr = open(filename)
    arrayOlines = fr.readlines()
    numberOfLines = len(arrayOlines)  # 得到文件行数
    returnMat = np.zeros((numberOfLines, 3))  # 创建返回的numpy矩阵
    classLabelVector = []  # 标签列表
    index = 0
    for line in arrayOlines:
        line = line.strip()  # 截取回车字符
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]  # 前三个为标签
        classLabelVector.append(listFromLine[-1])  # 最后一个为特征
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    '''
    归一化特征值
    :param dataSet:特征数据
    :return: 归一化特征，特征取值范围，特征最小值
    '''
    minVals = dataSet.min(0)  # 列最小值
    maxVals = dataSet.max(0)  # 列最大值
    ranges = maxVals - minVals
    # m = dataSet.shape[0]# 数据行数
    # normanDataSet = (dataSet - np.tile(minVals, (m, 1))) / (np.tile(maxVals, (m, 1)) - np.tile(minVals, (m, 1)))# 归一化结果
    normanDataSet = (dataSet - minVals) / (maxVals - minVals)
    return normanDataSet, ranges, minVals


def datingClassTest():
    '''
    分类器针对约会网站的测试代码
    :return: 准确率
    '''
    hoRatio = 0.1  # 测试数量占比
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')  # 读取数据
    normalDataSet, ranges, minVals = autoNorm(datingDataMat)  # 归一化特征
    m = datingDataMat.shape[0]
    numTestVecs = int(m * hoRatio)  # 下面使用range，需要强制类型转换为int
    errorCount = 0
    # 对每一条数据进行分类预测并统计错误数量
    for i in range(numTestVecs):
        classifierResult = classify0(normalDataSet[i, :], normalDataSet[numTestVecs:, :], datingLabels[numTestVecs:], 3)
        if (classifierResult != datingLabels[i]): errorCount += 1
    print('错误率为：%f' % ((errorCount) / numTestVecs))


def classifyPerson():
    '''
    根据输入值对结果进行预测
    :return: 预测结果
    '''
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTags = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent flier miles earned per year?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTags, iceCream])
    classifierResult = int(classify0((inArr - minVals) / ranges, normMat, datingLabels, 3))
    print('You will probably like this person:', resultList[classifierResult - 1])


def img2vector(filename):
    '''
    将32×32图像举证转换为1×1024向量
    :param filename: 文件
    :return: 转换后数组
    '''
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def handWritingClassTest():
    '''
    手写数字识别系统
    '''
    # 从文件名解析分类数字（训练集）
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    hwLabels = []
    for i in range(m):
        classNumStr = int(trainingFileList[i].split('.')[0].split('_')[0])# 获取标签
        trainingMat[i, :] = img2vector('trainingDigits/%s' % trainingFileList[i])
        hwLabels.append(classNumStr)
    # 从文件名解析分类数字（测试集）并统计预测错误数量
    testFileList = os.listdir('testDigits')
    mTest = len(testFileList)
    errorCount = 0
    for i in range(mTest):
        classNumStr = int(testFileList[i].split('.')[0].split('_')[0])# 获取标签
        vectorUnderTest = img2vector('testDigits/%s' % testFileList[i])
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        if (classNumStr != classifierResult): errorCount += 1
    print('错误率为：%f' % (errorCount / mTest))

# handWritingClassTest()
