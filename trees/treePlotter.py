import matplotlib.pyplot as plt

# 定义文本框和箭头格式
decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    '''
    绘制树节点
    :param nodeTxt: 注释内容
    :param centerPt: 注释位置
    :param parentPt: 被注释坐标点
    :param nodeType: 注释样式
    '''
    plt.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                                xytext=centerPt, textcoords='axes fraction',
                                va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)


# def createPlot():
#     # fig = plt.figure(1, facecolor='white')
#     # fig.clf()# 清空图形
#     plt.figure()
#     # plt.subplot(111, frameon=False)
#     plotNode('decisionNode', (0.5, 0.1), (0.1, 0.5), decisionNode)
#     plotNode('leafNode', (0.8, 0.1), (0.3, 0.8), leafNode)
#     plt.show()

def getNumLeafs(myTree):
    '''
    获取叶子节点数目
    :param myTree:
    :return:
    '''
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else: numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    '''
    获取树的层数
    :param myTree:
    :return:
    '''
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else: thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def retrieveTree(i):
    '''
    创建树
    :param i:
    :return:
    '''
    listOfTrees = [{'no surfacing':{0:'no', 1:{'flippers':
                                                   {0:'no', 1:'yes'}}}},
                   {'no surfacing':{0:'no', 1:{'flippers':
                                                   {0:{'head':{0:'no', 1:'yes'}}, 1:'no'}}}}
                   ]
    return listOfTrees[i]

def plotMidText(cntrPt, parentPt, txtString):
    '''
    文本信息位置
    :param cntrPt: 叶子节点位置
    :param parentPt: 树节点位置
    :param txtString: 文本信息
    '''
    xMid = (parentPt[0] + cntrPt[0]) / 2
    yMid = (parentPt[1] + cntrPt[1]) / 2
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    '''
    属性图各点的位置
    :param myTree:
    :param parentPt:
    :param nodeTxt:
    :return:
    '''
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1 + numLeafs)/2/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1/plotTree.totalD

def createPlot(inTree):
    '''
    绘图
    :param inTree:
    :return:
    '''
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = getNumLeafs(inTree)
    plotTree.totalD = getTreeDepth(inTree)
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1;
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()