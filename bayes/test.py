import re
# postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
#              ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
#              ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
#              ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
#              ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
#              ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
# classVec = [0,1,0,1,0,1]
#
# print(sum(classVec))

mySent = 'This book is the best book on python or M.L'
regEex = re.compile('[^a-zA-Z0-9]')
listOfTokens = mySent.split()
print(listOfTokens)

# print(list(range(50)))