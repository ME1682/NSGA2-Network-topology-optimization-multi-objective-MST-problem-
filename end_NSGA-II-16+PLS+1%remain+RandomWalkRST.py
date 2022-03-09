## 环境设定
import numpy as np
from deap import base, tools, creator, algorithms
import matplotlib.pyplot as plt
import copy
from collections import defaultdict
import random
import time
start=time.time()
N=16 #节点个数

# 问题定义
creator.create('MultiObjMin', base.Fitness, weights=(-1.0,-1.0,-1.0))  # -1.0代表最小化问题,1.0代表最大化问题
creator.create('Individual', list, fitness=creator.MultiObjMin)

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

#去重
def DelRepeat(data):
    new_data = [] # 用于存储去重后的list
    values = []   # 用于存储当前已有的值
    for d in data:
        if d.fitness.values not in values:
            new_data.append(d)
            values.append(d.fitness.values)
        elif d not in new_data:
            new_data.append(d)
            values.append(d.fitness.values)
    return new_data

#找不同
def different(data1,data2):
    new_data = [] # 用于存储去重后的list
    num=[]
    for d in data1:
        if d not in data2:
            new_data.append(d)
        else: num.append(data2.index(d))
    return new_data

#节点度中心性
degree=6
#return 0为符合度中心性要求
def degreecheck(ind,degree):
    nodeDict=generateSFromEdges(decoding(ind))
    flag=0
    for i in range(1,N+1):
        if len(nodeDict[str(i)]) > degree: flag=1
    return flag

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
    '''从给定的节点集合中以随机游走RST方法生成染色体
    输入：
    输出：ind -- 个体实数编码，长度为节点数-1
    '''
    nodeDict = generateSFromEdges(edges)#生成相邻节点集合，形如{'i':[j,k,l]}
    i=random.randint(1, N) #随机初始节点
    nodeSet = [i]  # 用于保存已经经过的节点
    ind=[]
    num=0
    while len(nodeSet) < N:
        num+=1
        j = nodeDict[str(i)][random.randint(0,len(nodeDict[str(i)])-1)]
        if j not in nodeSet:
            nodeSet.append(j)
            ind.append(str(min(i,j)) + ',' + str(max(i,j)))
        i = j #游走到下一个节点
        if num==10000:
            print(edges)
            print(nodeSet)
            print(ind)
    return ind

# 解码
def decoding(ind):
    '''对给定的染色体编码，解码为生成树(边的集合)
    输入：ind -- 个体实数编码，长度为节点数-1
    输出：generatedTree -- 边的集合，类似于edges，每个元素为形如'i,j'的str
    '''
    return ind

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
    A=1-normalization(7560.53,20418.66667, distenceSum)
    B=1-normalization(0, 10.526, disequilibriumSum)
    C=1-normalization(1, 15, jump)
    #mins = 3 - (A+B+C)
    #mins = 3 - (normalization(6247.9, 7000, distenceSum) + normalization(0, 16.522, disequilibriumSum) + normalization(1, 11, jump))
    return A,B,C

# 交叉操作
def cxPrimPred(ind1, ind2):
    '''给定两个个体，将其边叠加，再根据PrimPred编码方法生成新个体'''
    edges1 = decoding(ind1)  # 将个体解码为边
    edges2 = decoding(ind2)
    edgesCombined = list(set(edges1 + edges2))
    # 限制度数
    ind=PrimPredCoding(edges=edgesCombined)
    num=0
    while degreecheck(ind,degree)==1:
        ind = PrimPredCoding(edges=edgesCombined)
        num+=1
        if num==10:break
    return ind

# 突变操作
def mutLowestCost(ind):
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
    ''''# 选择两个点集中代价最小的边，添进边中
    minCostEdge = None
    minEdgeCost = 1e5
    for vert1 in nodeSet1:
        for vert2 in nodeSet2:
            key = str(min(vert1, vert2)) + ',' + str(max(vert1, vert2))
            if key in weightDict:
                if weightDict[key] < minEdgeCost:
                    minEdgeCost = weightDict[key]
                    minCostEdge = key
    edges = edges + [minCostEdge]'''
    #度中心性筛选
    nodeDict = generateSFromEdges(edges)
    num = [0]*(N+1)
    for i in range(1, N + 1):
        if str(i) not in nodeDict:
            num[i] = 0
        else: num[i] = len(nodeDict[str(i)])
        if num[i] >= degree:
            if i in nodeSet1:
                nodeSet1.remove(i)
            elif i in nodeSet2:
                nodeSet2.remove(i)
    # 随机选择一条边
    chooseEdge=edges[0]
    while chooseEdge in edges:
        a=nodeSet1[random.randint(0, len(nodeSet1)-1)]
        b=nodeSet2[random.randint(0, len(nodeSet2)-1)]
        if a != b:
            chooseEdge=(str(min(a,b))+','+str(max(a,b)))
    edges = edges + [chooseEdge]
    # 从边还原编码
    return PrimPredCoding(edges)

#find Neighborhood
def Neighborhood(Tree):
    nSet=[]
    candidates = toolbox.clone(edges)
    for edge in Tree:
        candidates.remove(edge)
    for edgeIdx in range(len(Tree)):
        u = int(Tree[edgeIdx].split(',')[0])
        v = int(Tree[edgeIdx].split(',')[1])
        TwoTrees = Tree[:edgeIdx] + Tree[edgeIdx + 1:]  # 删除选中的边
        # 将属于两个子图的顶点分别归如两个点集
        A = [0] * (len(ind) + 1)
        U = TwoTrees
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
        for edge in candidates:
            u = int(edge.split(',')[0])
            v = int(edge.split(',')[1])
            if (u in nodeSet1 and v in nodeSet2) or (v in nodeSet1 and u in nodeSet2):
                TwoTrees.append(edge)
                NewTree = toolbox.clone(TwoTrees)
                if NewTree not in nSet:
                    nSet.append(NewTree)
                TwoTrees.remove(edge)
        return nSet
#Pareto Local Search
def PLS(pop):
    if len(pop) == 0:
        return
    endpop = []
    PLSpop = toolbox.clone(pop)
    bestfit = 3.0
    num=0
    while len(PLSpop) != 0:
        num+=1
        PLSingpop=[]
        for ind in PLSpop:
            Tree = decoding(ind)
            Neighborhoods = Neighborhood(Tree)
            for nbr in Neighborhoods:
                fit = evaluate(PrimPredCoding(nbr))
                pop0 = toolbox.clone(pop[0])
                pop0[:] = PrimPredCoding(edges=nbr)
                pop0.fitness.values = tuple(fit)
                if sum(pop0.fitness.values) < hallOfFamefit[0]: #This is an interesting judge
                    if pop0 not in endpop:
                        PLSingpop.append(pop0)
                        if sum(pop0.fitness.values) < bestfit and degreecheck(pop0,degree) == 0:
                            bestfit = sum(pop0.fitness.values)
        PLSpop = toolbox.clone(PLSingpop)
        endpop = toolbox.clone(endpop + PLSpop)
        print('PLS'+str(num))
        if num==5:break
    print(bestfit)
    return endpop
# 注册工具
toolbox = base.Toolbox()
toolbox.register('individual', tools.initIterate, creator.Individual, PrimPredCoding)
toolbox.register('evaluate', evaluate)
toolbox.register('selectGen1', tools.selTournament, tournsize=2)
toolbox.register('select', tools.emo.selTournamentDCD) # 该函数是binary tournament，不需要tournsize
toolbox.register('mate', cxPrimPred)
toolbox.register('mutate', mutLowestCost)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
# 迭代数据
stats = tools.Statistics(key=lambda ind: ind.fitness.values)

##---------------------------
# 遗传算法参数
toolbox.ngen = 300
toolbox.popSize = 200
toolbox.cxpb = 0.6
toolbox.mutpb = 0.4
#cxpb:杂交概率 mutpb:突变概率 ngen:最大迭代代数 popSize:种群大小

# 迭代部分
# 第一代
pop = toolbox.population(toolbox.popSize) # 父代
fitnesses = toolbox.map(toolbox.evaluate, pop)
for ind, fit in zip(pop,fitnesses):
    ind.fitness.values = tuple(fit)
fronts = tools.emo.sortNondominated(pop, k=toolbox.popSize)# 将每个个体的适应度设置为pareto前沿的次序
for idx, front in enumerate(fronts):
    for ind in front:
        ind.fitness.values = (idx+1),
#存储历史最优解
Famesize=10
hallOfFame = toolbox.clone(pop[0:Famesize])
hallOfFamefit=[]
for i in range(Famesize):
    A = evaluate(hallOfFame[i])
    hallOfFamefit.append(3.0) #初始化
oldbest = toolbox.clone(hallOfFame[0])
oldbest_fit = 3.0
# 创建子代
offspring = toolbox.selectGen1(pop, toolbox.popSize,) # binary Tournament选择
offspring=toolbox.clone(offspring)#数据隔离
#offspring = algorithms.varAnd(offspring, toolbox, toolbox.cxpb, toolbox.mutpb)
for i in range(1,len(offspring)):#交叉
    if random.random() < toolbox.cxpb:
        offspring[i - 1][:] = copy.deepcopy(toolbox.mate(offspring[i - 1],offspring[i]))
        del offspring[i - 1].fitness.values

for i in range(len(offspring)):#变异
    if random.random() < toolbox.mutpb:
        offspring[i][:] = copy.deepcopy(toolbox.mutate(offspring[i]))
        del offspring[i].fitness.values
dif_origin=different(offspring,pop)
# 第二代之后的迭代
oldfit=[]
for gen in range(1, toolbox.ngen):
    combinedPop = pop + offspring # 合并父代与子代
    # 评价族群
    fitnesses = toolbox.map(toolbox.evaluate, combinedPop)
    for ind, fit in zip(combinedPop,fitnesses):
        ind.fitness.values = fit
    #去重
    combinedPop = DelRepeat(combinedPop)
    #dif=different(combinedPop,pop)
    # 快速非支配排序
    fronts = tools.emo.sortNondominated(combinedPop, k=len(combinedPop), first_front_only=False)
    # 拥挤距离计算
    for front in fronts:
        tools.emo.assignCrowdingDist(front)
    # 环境选择 -- 精英保留
    pop = []
    for front in fronts:
        pop += front
    pop = toolbox.clone(pop)
    pop = tools.selNSGA2(pop, k=toolbox.popSize, nd='standard')
    #更新历史最优解
    for i in range(len(pop)):
        if degreecheck(pop[i], degree) == 1:
            continue
        for j in range(Famesize):
            A = sum(pop[i].fitness.values)
            if(hallOfFame[j]==pop[i]):
                break
            elif(A < hallOfFamefit[j]):
                hallOfFame[j]=pop[i]
                hallOfFamefit[j]=A
                break
    oldfit.append(hallOfFamefit[0])
    #1%的额外精英保留机制
    for i in range(1):
        if hallOfFame[i] not in pop:
            pop.append(hallOfFame[i])
    # 创建子代
    while len(pop)%4 != 0:
        pop.append(pop[0])
    offspring = toolbox.select(pop, toolbox.popSize)
    offspring = toolbox.clone(offspring)
    #offspring = algorithms.varAnd(offspring, toolbox, toolbox.cxpb, toolbox.mutpb)
    for i in range(1, len(offspring), 2):  # 交叉
        if random.random() < toolbox.cxpb:
            offspring[i - 1][:]= toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values
    for i in range(len(offspring)):  # 变异
        if random.random() < toolbox.mutpb:
            offspring[i][:] = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
    print('第'+str(gen)+'轮')
# PLS搜索，照顾到时间复杂度，这里只对帕累托前沿搜索
combinedPop = toolbox.clone(PLS(fronts[0]) + pop)
# 去重
combinedPop = DelRepeat(combinedPop)
# dif=different(combinedPop,pop)
# 快速非支配排序
fronts = tools.emo.sortNondominated(combinedPop, k=len(combinedPop), first_front_only=False)
# 拥挤距离计算
for front in fronts:
    tools.emo.assignCrowdingDist(front)
# 环境选择 -- 精英保留
    pop = []
    for front in fronts:
        pop += front
        if len(pop) > toolbox.popSize:
            break
    pop = toolbox.clone(pop)
    pop = tools.selNSGA2(pop, k=toolbox.popSize, nd='standard')
#更新历史最优解
for i in range(len(pop)):
    if degreecheck(pop[i],degree)==1:
        continue
    for j in range(Famesize):
        A = sum(pop[i].fitness.values)
        if(hallOfFame[j]==pop[i]):
            break
        elif A < hallOfFamefit[j]:
            hallOfFame[j]=pop[i]
            hallOfFamefit[j]=A
            break
oldfit.append(hallOfFamefit[0])
## 输出结果
print('名人堂')
for i in range(Famesize):
    if degreecheck(hallOfFame[i],degree)==0:print('Yes')
    else:print('False')
    TestInd=hallOfFame[i]
    testEdges = decoding(TestInd)
    print('生成树的边为：' + str(testEdges))
    print(str(evaluate(TestInd))+' 和:' + str(sum(evaluate(TestInd))))

print("最终个体：")
endpop=[]
for j in range(1):
    for i in range(len(fronts[j])):
        if fronts[j][i] not in endpop:
            endpop.append(fronts[j][i])
ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
ax.set_title('end_pop')  # 设置本图名称
for i in range(len(endpop)):
    if degreecheck(endpop[i],degree)==0 : print('Yes')
    else:print('False')
    TestInd=endpop[i]
    testEdges = decoding(TestInd)
    eva=list(evaluate(TestInd))
    print('生成树的边为：' + str(testEdges))
    print(str(eva) +' 和:' + str(sum(eva)))
    ax.scatter(eva[0],eva[1],eva[2], c='r')  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色

end=time.time()
print('Running time: %s Seconds'%(end-start))

ax.set_xlabel('distance')  # 设置x坐标轴
ax.set_ylabel('disequilibrium')  # 设置y坐标轴
ax.set_zlabel('maxjump')  # 设置z坐标轴
plt.show()

bx = plt.subplot()
plt.plot(oldfit)
bx.set_xlabel('gen')  # 设置x坐标轴
bx.set_ylabel('fitness')  # 设置y坐标轴
plt.show()
print(oldfit)