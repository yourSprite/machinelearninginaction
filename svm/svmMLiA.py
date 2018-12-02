import numpy as np


def loadDataSet(fileName):
    '''
    获取特征和标签
    :param fileName:
    :return:
    '''
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    '''

    :param i: alpha下标
    :param m: alpha数目
    :return:
    '''
    j = i
    while (j == i):
        j = int(np.random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    '''
    调整大于H或小于L的alpha值
    :param aj: alpha值
    :param H: 最大值
    :param L: 最小值
    :return:
    '''
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    '''
    简化版SMO短发
    :param dataMatIn: 数据集
    :param classLabels: 类别标签
    :param C: 常数C
    :param toler: 容错率
    :param maxIter: 退出前最大迭代次数
    :return: 划分超平面参数
    '''
    dataMatrix = np.mat(dataMatIn);
    labelMat = np.mat(classLabels).transpose()
    b = 0;
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0  # 记录alpha是否进行优化
        for i in range(m):
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])  # 误差
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)  # 随机选择第二个alpha
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])  # 误差
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):  # 保证alpha在0与C之间
                    L = max(0, alphas[j] - alphas[i])
                    H = max(C, C + alphas[j] - alphas[i])

                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = max(C, alphas[j] + alphas[i])
                if L == H: print("L==H"); continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                      dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T  # alpha最有修改量
                if eta >= 0: print("eta>=0"); continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                # 设置常数项
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * \
                              dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * \
                              dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        '''
        建立一个数据结构来保存重要值
        :param dataMatIn: 输入数据集
        :param classLabels: 类别标签
        :param C: 常数C
        :param toler: 容错率
        '''
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = self.toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))  # 参数alpha
        self.b = 0  # 参数b
        self.eCache = np.mat(np.zeros(self.m, 2))  # 误差缓存，第一列为是否有效的标志位，第二列为E的实际值


def calcEk(oS, k):
    '''
    计算E值并返回
    :param oS: 上面创建的数据结构
    :param k: 序号k
    :return: 误差Ek
    '''
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.x[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):
    '''
    选择第二个alpha值
    :param i:
    :param oS:
    :param Ei:
    :return:
    '''
    maxk = -1;
    maxDeltaE = 0;
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]  # 返回E非零E值所对应的alpha值
    if len((validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i: continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxk = k;
                maxDeltaE = deltaE;
                Ej = Ek
        return maxk, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    '''
    计算误差并存入缓存中
    :param oS:
    :param k:
    :return:
    '''
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):
    '''
    完整版Platt SMO算法中的优化例程
    :param i:
    :param oS:
    :return:
    '''
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
            ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy();
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas.[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H: print("L==H"); return 0
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - \
              oS.X[j, :] * oS.X[j, :].T
        if eta >= 0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas(j), H, L)
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough");
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold-oS.alphas[j])
        updateEk(oS, j)
        b1 = oS.b - Ei -oS.labelMat[i]*(oS.alphas[i]-alphaIold)*\
             oS.X[i, :]*oS.X[i, :].T - oS.labelMat[j]*\
             (oS.alphas[j]-alphaJold)*oS.X[i, :]*oS.X[j, :].T
        b2 = oS.b - Ej -oS.labelMat[i]*(oS.alphas[i]-alphaIold)*\
             oS.X[i, :]*oS.X[j, :].T - oS.labelMat[j]*\
             (oS.alphas[j]-alphaJold)*oS.X[j, :]*oS.X[j, :].T
        if(0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif(0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0