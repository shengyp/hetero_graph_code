#!/usr/bin/env python
# coding: utf-8


'''
small_matHUAWEI.npy:
	{UA:array[], #　Adjacency matrix between different types of nodes
	 AT:array[],
	 ...}
'''
import numpy as np

ajd = np.load('small_matHUAWEI.npy')

ajd
ajd = ajd[()]

path1=['UA','AT','TA','AU']
path2=['UT','TA','AT','TU']
path3=['UT','TU']
path4=['UA','AU']

def metapath(path,adj):

    temp = adj[path[0]]
    for i in range(1,len(path)):
        temp = np.dot(temp,adj[path[i]])
    print(temp)
        
    h = temp.shape[0]
    newtemp = np.zeros((h,h))
    mid = (temp.max()+temp.min())/2
    print('mid:',mid)
    print('max:',temp.max())
    print('min:',temp.min())
    
    count = 0
    for a in range(h):
        for b in range(h):
			#if temp[a][b] > 1:
            if temp[a][b] > mid and a != b:
                newtemp[a][b] = 1
                count+=1
    print(newtemp)
    print(count)
        
    
    return newtemp


# In[14]:


UATAU = metapath(path1,ajd)


# In[15]:


UTATU = metapath(path2,ajd)


# In[16]:


UTU = metapath(path3,ajd)


# In[17]:


UAU = metapath(path4,ajd)


# In[18]:


UT_AU = np.multiply(UTU,UAU)
print(UT_AU)


# In[21]:


newadj_data = []
newadj_data.append(UATAU)
newadj_data.append(UTATU)
newadj_data.append(UTU)

np.save('adj_data.npy',newadj_data)













