#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# # ===========各个节点类别数量=============

# In[ ]:


init = pd.read_csv('IMDB.csv') 


# In[ ]:


init.head()


# In[ ]:


list(init['体裁']).count('Action')


# In[ ]:


list(init['体裁']).count('Drama')


# In[ ]:


list(init['体裁']).count('Comedy')


# 各类别数量：  
# Action：1013  
# Drama：1309  
# Comedy：1305

# # ==========生成节点编号文件=========

# In[ ]:


initdata = pd.read_csv('IMDB.csv')


director = list(set(initdata['导演']))


# 导演数量为1714

movie = list(set(initdata['电影']))


# 电影数量为3627


actor1 = list(set(initdata['演员1']))
actor2 = list(set(initdata['演员2']))
actor3 = list(set(initdata['演员3']))

actortemp = []
actortemp+=(actor1)
actortemp+=(actor2)
actortemp+=(actor3)

actor = list(set(actortemp))[1:]


# 演员数量为4340

rate = list(set(initdata['评级']))


# 评级数量为11

# In[ ]:


len(rate)


# In[ ]:


def output(concent, path):
    with open(path, 'w+') as fi:
        count = 0
        for li in concent:
            fi.write(str(li)+'\t'+str(count)+'\n')
            count+=1
    print('over output...')


# In[ ]:


output(rate,'rate.txt')
output(actor,'actor.txt')
output(movie,'movie.txt')
output(director,'director.txt')


# 导演：director, 1714    
# 电影：movie, 3627    
# 演员：actor, 4340  
# 评级：rate, 11    

# # ==============================================================

# 节点：导演（D），电影（M），演员（A），内容分级（R）  
# 
# 路径： （1）M-A-M-A-M；（2）M-A-D-A-M；（3）M-A-D(R)-A-M；（4）M-A-M(D)-A-M；  
# 
# 需要的文件：（1）MA；（2）DA；（3）RA；（4）Movie_label

# # ===========生成节点关系文件=============

# In[ ]:


def dic(path):
    name = {}
    with open(path,'r') as fi:
        data = fi.readlines()
        for da in data:
            dalist = da.strip('\n').split('\t')
            name[dalist[0]] = int(dalist[1])
    print('over output...')
    return name


# In[ ]:


director = dic('director.txt')
movie = dic('movie.txt')
actor = dic('actor.txt')
rate = dic('rate.txt')


# In[ ]:


label = {'Action':0, 'Drama':1, 'Comedy':2}


# alldata：[0]导演；[1]电影；[2]演员1；[3]演员2；[4]演员3；[5]内容评级；[6]电影类型；[7]内容情节

# In[ ]:


with open('IMDB.csv','r',encoding='utf-8') as fi:
    temp = fi.readline()
    alldata = fi.readlines()


# predata：[0]导演；[1]电影；[2][演员1,演员2,演员3]；[3]内容评级；[4]电影类型；

# In[ ]:


predata = []
for da in alldata:
    temp = []
    da = da.strip('\n').split(',')
    temp.append(da[0])
    temp.append(da[1])
    temp.append(da[2:5])
    temp.append(da[5])
    temp.append(da[6])
    predata.append(temp)


# 生成MA.txt文件

# In[ ]:


for data in predata:
    for ac in data[2]:
        if ac != '':
            with open('MA.txt','a+') as fi:
                fi.write(str(movie[data[1]])+'\t'+str(actor[ac])+'\n')


# 生成DA.txt文件

# In[ ]:


for data in predata:
    for ac in data[2]:
        if ac != '':
            with open('DA.txt','a+') as fi:
                fi.write(str(director[data[0]])+'\t'+str(actor[ac])+'\n')


# 生成RA.txt文件

# In[ ]:


for data in predata:
    for ac in data[2]:
        if ac != '':
            with open('RA.txt','a+') as fi:
                fi.write(str(rate[data[3]])+'\t'+str(actor[ac])+'\n')


# 生成label.txt文件

# In[ ]:


# movie_label
movlab = {} 
for data in predata:
    movlab[data[1]] = label[data[4]]


# In[ ]:


movkey = list(movie.keys())
for mk in movkey:
    with open('label.txt','a+') as fi:
        fi.write(str(movlab[mk])+'\n')


# # ==============================================================

# # 生成ont-hot label

# In[ ]:


import numpy as np


# In[ ]:


with open('label.txt', 'r') as fi:
    labeldata = fi.readlines()


# In[ ]:


labellist = []
for i in labeldata:
    labellist.append(int(i.strip('\n')))


# In[ ]:


onehot_label = np.zeros((len(labellist),len(set(labellist))))
for i in range(len(labellist)):
    onehot_label[i][labellist[i]] = 1

#np.save('one_hot_labels.npy',onehot_label)


# # ========================================================

# # 生成训练集和测试集

# In[ ]:


import numpy as np


# In[ ]:


with open('label.txt', 'r') as fi:
    labeldata = fi.readlines()


# In[ ]:


labellist = []
for i in labeldata:
    labellist.append(int(i.strip('\n')))


# In[ ]:


def get_index1(lst, item):
    return [index for (index,value) in enumerate(lst) if value == item]


# In[ ]:


label0 = get_index1(labellist,0)
label1 = get_index1(labellist,1)
label2 = get_index1(labellist,2)


# In[ ]:


def huafen(lis):
    n = int(len(lis)*0.2)
    temp = sorted(lis)
    test = temp[:n]
    train = temp[n:]
    
    return test, train


# In[ ]:


test0,train0 = huafen(label0)
test1,train1 = huafen(label1)
test2,train2 = huafen(label2)


# In[ ]:


test_idx = []
test_idx += test0
test_idx += test1
test_idx += test2
#np.save('test_idx.npy',sorted(test_idx))


# In[ ]:


train_idx = []
train_idx += train0
train_idx += train1
train_idx += train2
#np.save('train_idx.npy',sorted(train_idx))






