#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# **构建DA矩阵和AD矩阵**

# In[ ]:


numA = 4340
numD = 1714
numM = 3627
numR = 11


# # =====================================================

# In[ ]:


def mat(loadpath,A,B):
    with open(loadpath,'r') as fi:
        DA_data = fi.readlines()

    DA_set = {}
    for i in range(len(DA_data)):
        a,b = DA_data[i].strip('\n').split('\t')
        DA_set[int(a)] = int(b)

    DA = np.zeros((A,B))
    AD = np.zeros((B,A))

    for i in DA_set:
        DA[i][DA_set[i]] = 1
        AD[DA_set[i]][i] = 1

    return DA,AD


# In[ ]:


DA, AD = mat('DA.txt',numD,numA)


# In[ ]:


MA,AM = mat('MA.txt',numM,numA)


# In[ ]:


RA,AR = mat('RA.txt',numR,numA)


# In[ ]:


mat = {}
mat['DA']=DA
mat['AD']=AD
mat['MA']=MA
mat['AM']=AM
mat['RA']=RA
mat['AR']=AR


# In[ ]:


#np.save('mat.npy',mat)


# # ===============================================

# In[ ]:


metapath = [['MA','AM','MA','AM'],
        ['MA','AD','DA','AM']]


# In[ ]:


def iniadj(path):
    temp = mat[path[0]]
    for i in path[1:]:
        temp = np.dot(temp,mat[i])
    return temp


# In[ ]:


ini_adj = []
for i in metapath:
    ini_adj.append(iniadj(i))


# In[ ]:


np.array(ini_adj).shape


# In[ ]:


ARRA = np.dot(mat['AR'],mat['RA'])
ADDA = np.dot(mat['AD'],mat['DA'])
AMMA = np.dot(mat['AM'],mat['MA'])


# In[ ]:


ARDA = np.multiply(ARRA,ADDA)
AMDA = np.multiply(AMMA,ADDA)


# In[ ]:


ini_adj.append(np.dot(np.dot(mat['MA'],ARDA),mat['AM']))


# In[ ]:


ini_adj.append(np.dot(np.dot(mat['MA'],AMDA),mat['AM']))


# In[ ]:


#np.save('ini_adj.npy',ini_adj)


# In[ ]:


ini_adj[0]


# # ==================================================

# # 构建邻接矩阵

# In[ ]:


small_adj_data = []
for i in range(len(ini_adj)):
    meta = ini_adj[i]
    temp = np.zeros(meta.shape)
    for j in range(meta.shape[0]):
        for k in range(meta.shape[1]):
            if meta[j][k] > 0 and j != k:
                temp[j][k] = 1
    small_adj_data.append(temp)



for i in range(np.array(small_adj_data[0]).shape[0]):
    for j in range(np.array(small_adj_data[0]).shape[1]):
        if small_adj_data[0][i][j] != 0 and i !=j:
            print(i,j,small_adj_data[0][i][j])


#np.save('small_adj_data.npy',small_adj_data)


# # =====================================================

# # 构建相似度矩阵

# In[ ]:


def know_sim(M, i, j):
    broadness = M[i][i] + M[j][j]
    overlap = 2*M[i][j]
    if broadness == 0:
        return 0
    else:
        return overlap/broadness


def sim_matrix(adj):
    num = np.array(adj).shape[0]
    sim = np.zeros((num, num))
    for i in range(num - 1):
        for j in range(i+1, num):
            sim[i][j] = know_sim(adj, i, j)
            sim[j][i] = sim[i][j]
    return sim


# In[ ]:


sim_matrix_adj = []
for adj in ini_adj:
    print('开始了一个新的meta-path！')
    sim_matrix_adj.append(sim_matrix(adj))


#np.save('sim_matrix.npy',sim_matrix_adj)







