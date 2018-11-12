from numpy import *


##CART算法的实现代码
#导入数据
def loadDataSet(fileName):
    dataMat = []                #假设最后一列是目标值
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #将每行映射成浮点数
        dataMat.append(fltLine)
    return dataMat

#切分得到两个子集
def binSplitDataSet(dataSet, feature, value):
    """
    将数据集合切分得到两个子集
    :param dataSet: 数据集合
    :param feature: 待切分的特征
    :param value: 该特征的某个值
    :return: 返回两个子集
    """
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]#数据集中第feature列的值大于value的分为一组
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]#数据集中第feature列的值小于等于value的分为一组
    return mat0,mat1

#回归树的叶节点生成函数
def regLeaf(dataSet):#returns the value used for each leaf
    return mean(dataSet[:,-1])#在回归树中，返回目标量的均值

#误差估计函数
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]#总方差=方差*数据集中样本的个数

#回归树的切分函数
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    """
    找到数据的最佳二元切分方式
    :param dataSet: 数据集
    :param leafType: 对创建叶节点的函数的引用
    :param errType: 对误差估计函数的引用
    :param ops: 是一个用户定义的参数构成的元组，用于控制函数的停止时机
    :return: 返回特征编号和切分特征值
    """
    tolS = ops[0]#容许的误差下降值
    tolN = ops[1]#切分的最少样本数
    #如果所有目标变量都是相同的值则退出
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #tolist()是将数组或矩阵转换为列表，set() 函数创建一个无序不重复元素集
        return None, leafType(dataSet) #返回None并同时产生叶节点
    m,n = shape(dataSet)
    #最佳切分也就是使得切分后能达到最低误差的切分
    S = errType(dataSet)#误差
    bestS = inf#正无穷
    bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex].T.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)#切分得到两个数据集
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue#如果某个子集的大小小于用户定义的参数tolN，则跳出本次循环，继续下一轮循环
            newS = errType(mat0) + errType(mat1)#新误差
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #如果下降(S-bestS)小于阈值tolS，则不要切分而直接创建叶节点
    if (S - bestS) < tolS:
        return None, leafType(dataSet) #返回None并同时产生叶节点
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)#切分得到两个数据集
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #如果切分出的数据集的大小小于用户定义的参数tolN
        return None, leafType(dataSet)#返回None并同时产生叶节点
    return bestIndex,bestValue#返回切分特性和特征值

#构建回归树
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#假设数据集是NumPy Mat，那么我们可以数组过滤
    """
    构建树
    :param dataSet: 数据集
    :param leafType: 建立叶节点的函数
    :param errType: 误差计算函数
    :param ops: 是一个用户定义的参数构成的元组，用于控制函数的停止时机
    :return: 存放树的数据结构的字典
    """
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#选择最佳分割，二元切分
    if feat == None: return val #如果切分达到停止条件，返回特征值
    retTree = {}#字典
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)#切分得到两子集
    retTree['left'] = createTree(lSet, leafType, errType, ops)#左子树
    retTree['right'] = createTree(rSet, leafType, errType, ops)#右子树
    return retTree#存放树的数据结构的字典



