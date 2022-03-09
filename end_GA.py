## 环境设定
import numpy as np
import matplotlib.pyplot as plt
from deap import base, tools, creator, algorithms
import random
from collections import defaultdict

params = {
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'legend.fontsize': 'small'
}
plt.rcParams.update(params)
N=16 #节点个数
from copy import deepcopy

# 问题定义
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))  # 最小化问题
creator.create('Individual', list, fitness=creator.FitnessMin)

#定义无人机能量与平均能量
E=[50,40,90,70,60,52,67,35,69,87,48,75,57,43,73,86]
E2=np.mat(np.zeros((N,N)))
for i in range(N):
    for j in range(N):
        E2[i,j]=(E[i]+E[j])/2 #定义链路能量 0 ~ N-1

# 个体编码：连通边
edges = [
    '1,2', '1,3', '1,4', '1,5', '1,6', '1,7', '1,8', '1,9', '1,10', '1,11', '1,12','1,13','1,14','1,15','1,16',
    '2,3', '2,4', '2,5', '2,6', '2,7', '2,8', '2,9', '2,10', '2,11', '2,12','2,13','2,14','2,15','2,16',
    '3,4', '3,5', '3,6', '3,7', '3,8', '3,9', '3,10', '3,11', '3,12','3,13','3,14','3,15','3,16',
    '4,5', '4,6', '4,7', '4,8', '4,9', '4,10', '4,11', '4,12','4,13','4,14','4,15','4,16',
    '5,6', '5,7', '5,8', '5,9', '5,10', '5,11', '5,12','5,13','5,14','5,15','5,16',
    '6,7', '6,8', '6,9', '6,10', '6,11', '6,12','6,13','6,14','6,15','6,16',
    '7,8', '7,9', '7,10', '7,11', '7,12','7,13','7,14','7,15','7,16',
    '8,9', '8,10', '8,11', '8,12','8,13','8,14','8,15','8,16',
    '9,10', '9,11', '9,12','9,13','9,14','9,15','9,16',
    '10,11', '10,12','10,13','10,14','10,15','10,16',
    '11,12','11,13','11,14','11,15','11,16',
    '12,13','12,14','12,15','12,16',
    '13,14','13,15','13,16',
    '14,15','14,16',
    '15,16'
]
#节点位置
position={'1':(300,1600),'2':(0,1200),'3':(600,1200),'4':(800,800),'5':(1500,1600),'6':(1200,1200),'7':(1800,1200),
          '8':(2000,800),'9':(300,400),'10':(0,0),'11':(600,0),'12':(800,-400),'13':(1500,400),'14':(1200,0),
          '15':(1800,0),'16':(2000,-400)
}

# 边的距离
weightDict={}
for i in range(1,N+1):
    for j in range(i,N+1):
        distence=(int)(((position[str(i)][0]-position[str(j)][0]) ** 2 + (position[str(i)][1]-position[str(j)][1]) ** 2) ** 0.5)
        weightDict[str(i)+','+str(j)]=distence

#节点度中心性
degree=4

#边生成点集
def generateSFromEdges(edges):
    '''用关联表存储图，从提供的边集中生成与各个节点i相邻的节点集合Si
    输入：edges -- list, 其中每个元素为每个节点上的边，每个元素均为一个str 'i,j'
    输出：nodeDict -- dict, 形如{'i':[j,k,l]}，记录从每个节点能到达的其他节点
    '''
    nodeDict = {}
    for edge in edges:
        i, j = edge.split(',')
        if not i in nodeDict:
            nodeDict[i] = [int(j)]
        else:
            nodeDict[i].append(int(j))
        # 无向图中(i,j)与(j,i)是相同的
        if not j in nodeDict:
            nodeDict[j] = [int(i)]
        else:
            nodeDict[j].append(int(i))
    return nodeDict

#生成从节点i出发的所有边
def eligibleEdgeSet(nodeDict, i):
    '''辅助函数，生成从节点i出发的所有边(i,j)
    输入：nodeDict -- dict，记录每个节点能到达的其他节点
    i -- 起始节点，int
    输出：edgeSet -- list，记录从节点i出发可能的所有边的集合，其中每个元素为一条边，形如'i,j'的str
    '''
    endNodeSet = nodeDict[str(i)]  # i节点的所有后续节点
    edgeSet = []
    for eachNode in endNodeSet:
        edgeSet.append(str(i) + ',' + str(eachNode))
    return edgeSet

#给定一个noseSet，返回其中所有可能的边
def genEdgeFromNodeSet(nodeSet):
    '''辅助函数，给定一个noseSet，返回其中所有可能的边
    输入： nodeSet -- list，每个元素均为int,代表一个节点
    输出：edgesGen -- list, 每个元素代表一条边，形如'i,j'的str
    '''
    from itertools import combinations
    combs = combinations(nodeSet, 2)
    edgesGen = []
    for eachItem in combs:
        edgesGen.append(str(eachItem[0]) + ',' + str(eachItem[1]))
        edgesGen.append(str(eachItem[1]) + ',' + str(eachItem[0]))
    return edgesGen

#从给定的节点集合中以PrimPred方法生成染色体
def PrimPredCoding(edges=edges):
    '''从给定的节点集合中以PrimPred方法生成染色体
    输入：
    输出：ind -- 个体实数编码，长度为节点数-1
    '''
    nodeDict = generateSFromEdges(edges)#生成相邻节点集合，形如{'i':[j,k,l]}
    nodeCount = len(nodeDict)  # 这个长度等于节点数
    i = 1
    nodeSet = [i]  # 用于保存迭代中间变量
    edgeSet = eligibleEdgeSet(nodeDict, i)  # 从i出发所有可能的边
    iterIdx = 1
    ind = [0] * (nodeCount - 1)  # [1]作为默认起始点，需要的编码长度为节点数-1
    while iterIdx < nodeCount:
        edgeSelected = edgeSet[random.randint(0, len(edgeSet) - 1)]  # 随机选取一条可行边
        i = int(edgeSelected.split(',')[0])  # 所选边的起点
        j = int(edgeSelected.split(',')[1])  # 所选边的终点，范围为2到len(nodeDict)+1
        #print(len(ind), j-1)
        ind[j - 2] = i  # 注意j是从1到len(#node)的，作为index应该减去1
        nodeSet.append(j)
        i = j
        if not i == len(nodeDict) + 1:  # 当i时最终节点时，没有可用的边了
            edgeSet = edgeSet + eligibleEdgeSet(nodeDict, i)
        edgesToExclude = genEdgeFromNodeSet(nodeSet)  # 需要从集合中删掉的边
        edgeSet = list(set(edgeSet) - set(edgesToExclude))
        iterIdx += 1
    return ind


toolbox = base.Toolbox()
toolbox.register('individual', tools.initIterate, creator.Individual, PrimPredCoding)


# 解码
def decoding(ind):
    '''对给定的染色体编码，解码为生成树(边的集合)
    输入：ind -- 个体实数编码，长度为节点数-1
    输出：generatedTree -- 边的集合，类似于edges，每个元素为形如'i,j'的str
    '''
    generatedTree = []
    geneLen = len(ind)
    for i, j in zip(ind, range(2, 2 + geneLen)):
        generatedTree.append(str(min(i, j)) + ',' + str(max(i, j)))
    return generatedTree

#最大跳数
class jumpSolution(object):
    def treeDiameter(self, edges):
        """求树的直径，dfs
        :type edges: List[List[int]]
        :rtype: int
        """
        if not edges:
            return 0
        self.neibors = defaultdict(set)
        self.res = 0
        self.seen=set()
        for edge in edges:  # 建树
            cnt=edge.split(',')
            self.neibors[int(cnt[0])].add(int(cnt[1]))
            self.neibors[int(cnt[1])].add(int(cnt[0]))
        def getHeight(seen,node):
            res = []
            for neibor in self.neibors[node]:
                if not neibor in seen:
                    seen.add(neibor)
                    res.append(getHeight(seen,neibor))
            while len(res) < 2:  # 如果孩子少于两个，就给它补空的上去
                res.append(0)
            res = sorted(res)
            self.res = max(self.res, sum(res[-2:]))  # 取最长的两个子树长度
            return 1 + max(res)
        getHeight(self.seen,1)
        return self.res

def maxjump(ind):
    '''寻找个体ind解码后的生成树的最大跳数'''
    generatedTree = decoding(ind)
    S=jumpSolution()
    jump=S.treeDiameter(generatedTree)
    return jump

#不均衡度
def disequilibrium(ind):
    '''计算链路能量的不均衡度'''
    generatedTree = decoding(ind)
    E_average = 0 #平均能量
    E_disequilibrium = 0 #能量不均衡度
    for edge in generatedTree:
        E3 = edge.split(',')
        E_average = E_average + E[int(E3[0]) - 1] + E[int(E3[1]) - 1]
    E_average = E_average/2/(N-1)
    for edge in generatedTree:
        E3 = edge.split(',')
        E_disequilibrium = E_disequilibrium + (E2[int(E3[0]) - 1, int(E3[1]) - 1] - E_average) ** 2
    E_disequilibrium = (E_disequilibrium/(N-1))**0.5
    return E_disequilibrium

#归一化函数
def normalization(a,b,f):
    '''a为下限，b为上限'''
    if f<=a : s=1
    elif f>a and f<b : s=(b-f)/(b-a)
    else: s=0
    return s

#评估函数
def evaluate(ind):
    '''对给定的染色体编码，返回给定边的权值之和'''
    generatedTree = decoding(ind)
    distenceSum = 0
    for eachEdge in generatedTree:
        distenceSum += weightDict[eachEdge]
    disequilibriumSum=disequilibrium(ind)
    jump=maxjump(ind)
    #最优解
    #mins=3-(normalization(6247.9,10000,distenceSum) + normalization(0,9.193,disequilibriumSum) + normalization(1,7,jump))
    # 一字长蛇阵
    A=normalization(7560.53,20418.66667, distenceSum)
    B=normalization(0, 10.526, disequilibriumSum)
    C=normalization(1, 15, jump)
    mins = 3 - (A+B+C)
    #mins = 3 - (normalization(6247.9, 7000, distenceSum) + normalization(0, 16.522, disequilibriumSum) + normalization(1, 11, jump))
    return mins

# 交叉操作：可能会出现环，但是没有做处理，疑问
def cxPrimPred(ind1, ind2):
    '''给定两个个体，将其边叠加，再根据PrimPred编码方法生成新个体'''
    edges1 = decoding(ind1)  # 将个体解码为边
    edges2 = decoding(ind2)
    edgesCombined = list(set(edges1 + edges2))
    return PrimPredCoding(edges=edgesCombined)


# 突变操作
def mutLowestCost(ind, weightDict=weightDict):
    '''给定一个个体，用lowest cost method生成新个体，先将父代染色体中随机删除一条边，将原图
    分为两个互不连通的子图，然后选择连接这两个子图的具有最小权数的边并连接子图'''
    # 将原图分为两个互不连通的子图
    edges = decoding(ind)
    edgeIdx = random.randint(0, len(edges) - 1)  # 选择一条需要删除的边
    u = int(edges[edgeIdx].split(',')[0])
    v = int(edges[edgeIdx].split(',')[1])
    edges = edges[:edgeIdx] + edges[edgeIdx + 1:]  # 删除选中的边
    # 将属于两个子图的顶点分别归如两个点集
    A = [0] * (len(ind) + 1)
    U = edges
    while U:
        randomEdgeIdx = random.randint(0, len(U) - 1)  # 随机选择一条边(i,j)
        i = int(U[randomEdgeIdx].split(',')[0])
        j = int(U[randomEdgeIdx].split(',')[1])
        U = U[:randomEdgeIdx] + U[randomEdgeIdx + 1:]  # 删除选中的边
        if A[i - 1] == 0 and A[j - 1] == 0:
            l = min(i, j)
            A[i - 1] = l
            A[j - 1] = l
        elif A[i - 1] == 0 and A[j - 1] != 0:
            A[i - 1] = A[j - 1]
        elif A[i - 1] != 0 and A[j - 1] == 0:
            A[j - 1] = A[i - 1]
        else:
            if A[i - 1] < A[j - 1]:
                idx = [A[_] == A[j - 1] for _ in range(len(A))]
                A = np.where(idx, A[i - 1], A)
            elif A[i - 1] > A[j - 1]:
                idx = [A[_] == A[i - 1] for _ in range(len(A))]
                A = np.where(idx, A[j - 1], A)
    nodeSet1 = [_ + 1 for _ in range(len(A)) if A[_] == A[u - 1]]  # 注意index和节点编号的关系
    nodeSet2 = [_ + 1 for _ in range(len(A)) if A[_] == A[v - 1]]
    # 选择两个点集中代价最小的边，添进边中
    minCostEdge = None
    minEdgeCost = 1e5
    for vert1 in nodeSet1:
        for vert2 in nodeSet2:
            key = str(min(vert1, vert2)) + ',' + str(max(vert1, vert2))
            if key in weightDict:
                if weightDict[key] < minEdgeCost:
                    minEdgeCost = weightDict[key]
                    minCostEdge = key
    edges = edges + [minCostEdge]
    # 从边还原编码
    return PrimPredCoding(edges)


# 注册工具
toolbox.register('evaluate', evaluate)
toolbox.register('select', tools.selTournament, tournsize=2)
toolbox.register('mate', cxPrimPred)
toolbox.register('mutate', mutLowestCost)

# 迭代数据
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register('min', np.min)
stats.register('avg', np.mean)
stats.register('std', np.std)

##---------------------------
# 遗传算法参数
toolbox.ngen = 300
toolbox.popSize = 200
toolbox.cxpb = 0.6
toolbox.mutpb = 0.4
#cxpb:杂交概率 mutpb:突变概率 ngen:最大迭代代数 popSize:种群大小

# 生成初始族群
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
pop = toolbox.population(toolbox.popSize)

# 遗传算法主程序
hallOfFame = tools.HallOfFame(maxsize=100)
logbook = tools.Logbook()
logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

# Evaluate the individuals with an invalid fitness
invalid_ind = [ind for ind in pop if not ind.fitness.valid]
fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
for ind, fit in zip(invalid_ind, fitnesses):
    ind.fitness.values = tuple([fit]) #这里需要输入元组值，tuple只能转化list

hallOfFame.update(pop)

record = stats.compile(pop) if stats else {}
logbook.record(gen=0, nevals=len(invalid_ind), **record) #日志

# Begin the generational process
for gen in range(1, toolbox.ngen + 1):
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))

    # Vary the pool of individuals
    for i in range(1, len(offspring), 2):
        if random.random() < toolbox.cxpb:
            offspring[i - 1][:] = toolbox.mate(offspring[i - 1],
                                               offspring[i])
            del offspring[i - 1].fitness.values

    for i in range(len(offspring)):
        if random.random() < toolbox.mutpb:
            offspring[i][:] = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = tuple([fit])

    # Update the hall of fame with the generated individuals
    hallOfFame.update(offspring)
    print(str(evaluate(hallOfFame[0])),end=',')
    # Replace the current population by the offspring
    pop[:] = offspring

    # Append the current generation statistics to the logbook
    record = stats.compile(pop) if stats else {}
    logbook.record(gen=gen, nevals=len(invalid_ind), **record)
print(logbook)

## 输出结果
bestInd = hallOfFame.items[0]
bestFitness = bestInd.fitness.values
bestEdges = decoding(bestInd)
print('最小生成树的边为：' + str(bestEdges))
print('最小生成树的代价为：' + str(bestFitness))
print('名人堂：')
for i in range(30):
    TestInd=hallOfFame.items[i]
    testEdges = decoding(TestInd)
    print('生成树的边为：' + str(testEdges) + str(evaluate(TestInd)))

print("最终个体：")
for i in range(100):
    TestInd=pop[i]
    testEdges = decoding(TestInd)
    print('生成树的边为：' + str(testEdges) + str(evaluate(TestInd)))
## 画出迭代图
minFit = logbook.select('min')
avgFit = logbook.select('avg')
plt.plot(minFit, 'b-', label='Minimum Fitness')
plt.plot(avgFit, 'r-', label='Average Fitness')
plt.xlabel('# Gen')
plt.ylabel('Fitness')
plt.legend(loc='best')