import trees
import treePlotter
from math import log

# dataSet = [[1, 1, 'yes'],
#                [1, 1, 'yes'],
#                [1, 0, 'no'],
#                [0, 1, 'no'],
#                [0, 1, 'no']]
# numEntries = len(dataSet)
# labelCounts = {}
# # 取出标签并统计出现次数保存在字典中
# for featVec in dataSet:
#     currentLabel = featVec[-1]
#     if currentLabel not in labelCounts.keys():
#         labelCounts[currentLabel] = 0
#     labelCounts[currentLabel] += 1
# # 计算标签的香农商
# shannonEnt = 0
# for key in labelCounts:
#     prob = labelCounts[key] / numEntries
#     shannonEnt -= prob * log(prob, 2)
#
# print(shannonEnt)

# axis = 0
# value = 1
# retDataSet = []
# for featVec in dataSet:
#     if featVec[axis] == value:
#         reducedFeatSet = featVec[:axis]
#         reducedFeatSet.extend(featVec[axis+1:])
#         retDataSet.append(reducedFeatSet)
# print(retDataSet)

# numFeatures = len(dataSet[0]) - 1
# baseEntropy = trees.calcShannonEnt(dataSet)
# bestInfoGain = 0; bestFeature = -1
# # 遍历所有特征
# for i in range(numFeatures):
#     # 遍历所有特征可能的值
#     featList = [example[i] for example in dataSet]
#     uniqueVals = set(featList)
#     newEntropy = 0
#     # 遍历属性，对每一个属性划分数据集
#     for value in uniqueVals:
#         subDataSet = trees.splitDataSet(dataSet, i, value)
#         # 计算新划分数据集的熵值并进行求和
#         prob = len(subDataSet) / len(dataSet)
#         newEntropy += prob * trees.calcShannonEnt(subDataSet)
#     infoGain = baseEntropy - newEntropy
#     if(infoGain > bestInfoGain):
#         bestInfoGain = infoGain
#         bestFeature = i
# print(bestInfoGain)
# print(bestFeature)

# dict = {'测试1':1, '测试2':2}
# print(type(dict.keys()))

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = trees.createTree(lenses, lensesLabels)
# print(lensesTree)
treePlotter.createPlot(lensesTree)
