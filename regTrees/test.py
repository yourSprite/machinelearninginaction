import regTrees
import numpy as np

# testMat = np.mat(np.eye(4))
# mat0, mat1 = regTrees.binSplitDataSet(testMat, 1, 0.5)
# print(type(mat0))
# print(type(mat1))

# myDat = regTrees.loadDataSet('ex00.txt')
# myDat = np.mat(myDat)
# print(regTrees.createTree(myDat))

# myDat1 = regTrees.loadDataSet('ex0.txt' )
# myDat1 = np.mat(myDat1)
# print(regTrees.createTree(myDat1))

# myDat2 = regTrees.loadDataSet('ex2.txt')
# myDat2 = np.mat(myDat2)
# print(regTrees.createTree(myDat2))

# myTree = regTrees.createTree(myDat2, ops=(0, 1))
# myDatTest = regTrees.loadDataSet('ex2test.txt')
# myMat2Test = np.mat(myDatTest)
# tree = regTrees.prune(myTree, myMat2Test)
# print(tree)

# myMat2 = np.mat(regTrees.loadDataSet('exp2.txt'))
# tree = regTrees.createTree(myMat2, regTrees.modelLeaf, regTrees.modelErr, (1, 10))
# print(type(tree))

# trainMat = np.mat(regTrees.loadDataSet('bikeSpeedVsIq_train.txt'))
# testMat = np.mat(regTrees.loadDataSet('bikeSpeedVsIq_test.txt'))
# myTree = regTrees.createTree(trainMat, ops=(1, 20))
# yHat = regTrees.createForeCast(myTree, testMat[:, 0])
# print(np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])

