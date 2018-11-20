import numpy as np

def loadDataSet():
    '''
    创建样本
    :return:
    '''
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1代表侮辱性文字，0代表正常言论
    return postingList, classVec

def createVocabList(dataSet):
    '''
    收集文档中词的集合
    :param dataSet:
    :return:
    '''
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)# 取并集
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    '''
    将词是否出现转换为向量
    :param vocabList: 判断词汇列表
    :param inputSet: 文档出现词汇
    :return:
    '''
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print('the word:%s is not in my Vocabulary!' % word)
    return returnVec

def bagOfWords2VecMN(vocabList, inputSet):
    '''
    词袋模型，统计单词出现次数
    :param vocabList:
    :param inputSet:
    :return:
    '''
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        returnVec[vocabList.index(word)] += 1
    else: print('the word:%s is not in my Vocabulary!' % word)
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    '''
    朴素贝叶斯分类器训练函数，得到贝叶斯函数用于计算的概率
    :param trainMatrix:
    :param trainCategory:
    :return:
    '''
    numTarinDocs = len(trainMatrix)# 文档数量
    numWOrds = len(trainMatrix[0])# 每个文档中的词汇数量
    pAbusive = sum(trainCategory)/numTarinDocs# 侮辱性文档概率
    # 初始化概率
    # p0Num = np.zeros(numWOrds); p1Num = np.zeros(numWOrds)
    p0Num = np.ones(numWOrds); p1Num = np.ones(numWOrds)
    # p0Denom = 0; p1Denom = 0
    p0Denom = 2; p1Denom = 2
    for i in range(numTarinDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]# 增加词条计数值
            p1Denom += sum(trainMatrix[i])# 增加所有词条计数值
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # p1Vect = p1Num / p1Denom
    # p0Vect = p0Num / p0Denom
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''
    朴素贝叶斯分类函数
    :param vec2Classify: 要分类的向量
    :param p0Vec:
    :param p1Vec:
    :param pClass1:
    :return:
    '''
    p1 = sum(vec2Classify*p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify*p0Vec) + np.log(1-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
def testingNB():
    '''
    测试函数
    :return:
    '''
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))

def textParse(bigString):
    '''
    将大写字符串解析为小写字符串列表并去掉长度小于2的字符串
    :param bigString: 输入文本
    :return: 字符串列表
    '''
    # listOfTokens = bigString.split()
    import re
    listOfTokens = re.split('[^a-zA-Z0-9]', bigString)  # 以除单词、数字外的任意字符串为分隔符
    # print(listOfTokens)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    '''
    利用交叉验证测试朴素贝叶斯错误率
    '''
    docList = []; classList = []; fullText= [];
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50)); testSet = []
    for i in range(10):
        # 随机构建训练集
        randIndex = int(np.random.uniform(0, len(trainingSet)))# 随机生成实数，范围(0, len)
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])# 从测试集中去掉
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        # 进行测试并统计错误率
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the errored rate is:', errorCount/len(testSet))