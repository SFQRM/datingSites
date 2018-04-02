"""
    # 示例：使用k-临近算法改进约会网站的配对效果 #
"""

from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

"""
    实施kNN分类算法
"""
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]      # shape方法是numpy的函数，shape[0]是第二维的长度
    # 距离计算
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet      # tile(A,n)函数是numpy的函数，功能是将数组A重复n次
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)     # 没有axis参数表示全部相加，axis＝0表示按列相加，axis＝1表示按照行的方向相加
    distances = sqDistances**0.5
    # 距离计算
    sortedDistIndicies = distances.argsort()        # argsort()函数是numpy的函数，函数返回的数组值从小到大的索引值
    classCount = {}     # 初始化空字典
    # 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # dict[k] = v 将值v关联到键k上
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1       # 字典get()函数返回指定键的值，如果值不在字典中返回默认值。
    # 选择距离最小的k个点
    # classCount.iteritems()是将classCount字典分解为元组列表
    # sorted()方法返回的是一个新的列表
    sortedClassCount = sorted(classCount.items(),           # 要排序的对象
                              key=operator.itemgetter(1),   # 指定取待排序元素的第二项进行排序
                              reverse=True)                 # 降序
    return sortedClassCount[0][0]

"""
    step1——收集数据：略
    step2——准备数据：从文本文件中解析数据
"""
# 将文本记录转换成为NumPy的解析程序
def file2matrix(filename):
    fr = open(filename)
    """ .readlines()函数自动将文件内容分析成一个行的列表。遇到回车分隔
        .readline()函数每次只读取一行 """
    arrayOLines = fr.readlines()
    """
        len()是list的方法，返回列表元素个数 """
    numberOfLines = len(arrayOLines)        # 得到文件行数
    """ numpy函数，zeros函数返回一个给定形状和类型的用0填充的数组
                用法：zeros(shape, dtype=float, order='C') """
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []       # 初始化一个空列表，用于存放标签向量
    index = 0
    for line in arrayOLines:
        line = line.strip()     # 截取掉字符串开头和末尾的空白
        listFromLine = line.split('\t')     # 使用tab字符将整行数据分割成一个元素列表
        """
            arrayName[x, :]代表选取数组第x-1行
            arrayName[: ,x]代表选取数组第x-1列 """
        returnMat[index, :] = listFromLine[0:3]     # 选取前3个元素，将他们存储到特征矩阵中
        classLabelVector.append(int(listFromLine[-1]))      # 将列表的最后一列存储到向量cLV中
        index += 1
    return returnMat, classLabelVector

datingDateMat,datingLabels = file2matrix('datingTestSet2.txt')
# print(datingDateMat, '\n', datingLabels)

"""
    step2——准备数据：从文本文件中解析数据
"""
# 归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(axis=0)       # 每列(每个特征值)最小值
    maxVals = dataSet.max(0)        # 每列(每个特征值)最大值
    ranges = maxVals - minVals      # 求区间长度
    normDataSet = zeros(shape(dataSet))     # 构造一个全0矩阵（数组）
    m = dataSet.shape[0]        # 求数据集的行数
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals

"""
    step3——分析数据：使用Matplotlib创建散点图
"""
fig = plt.figure()      # 建立空白图（画布）
ax = fig.add_subplot(111)       # 画子图，参数111代表将画布分为1行1列，图像在图从左到右从上到下的第1块
ax.scatter(datingDateMat[:,0], datingDateMat[:,1],
           15.0*array(datingLabels), 15.0*array(datingLabels))      # 画出datingDataMat矩阵的第二列（游戏百分比）和第三列（冰淇淋公斤数）数据
plt.show()

"""
    step4——训练算法：略
    step5——测试算法：作为完整程序验证分类器
"""
# 分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.10
    datingDateMat, datingLabels = file2matrix('datingTestSet2.txt')      # 读取约会网站数据
    normMat, ranges, minVals = autoNorm(datingDateMat)      # 归一化数据
    m = normMat.shape[0]        # 求归一化后数据矩阵的行数
    numTestVecs = int(m*hoRatio)        # 测试向量的数量
    errorCount = 0.0        # 分类错误累加器
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],                          # 输入向量
                                     normMat[numTestVecs:m,:],              # 训练样本集，选取从第numTestVecs行到m-1行
                                     datingLabels[numTestVecs:m],           # 特征向量
                                     3)                                     # 最近临近数
        print("the classifier came back with：%d, the real answer is：%d",
              classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is:",errorCount/float(numTestVecs)*100,"%")

# print(datingClassTest())

"""
    step6——使用算法：构建完整可用系统
"""
# 约会网站预测函数
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing vedio games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDateMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDateMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges,
                                 normMat,
                                 datingLabels,
                                 3)
    print("You will probably like this person：",
          resultList[classifierResult - 1])

"""
    执行
"""
print(classifyPerson())
