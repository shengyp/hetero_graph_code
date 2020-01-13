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
	
	return temp

UATAU = metapath(path1,ajd)
UTATU = metapath(path2,ajd)
UTU = metapath(path3,ajd)
UAU = metapath(path4,ajd)
UT_AU = np.multiply(UTU,UAU)

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
			sim[i][j] = know_sim(M, i, j)
			sim[j][i] = sim[i][j]

UATAUsim = sim_matrix(UATAU)
UTATUsim = sim_matrix(UTATU)
UT_AUsim = sim_matrix(UT_AU)

sim_mat = []
sim_mat.append(UATAUsim)
sim_mat.append(UTATUsim)
sim_mat.append(UT_AUsim)

np.save('sim_mat.npy', sim_mat)
	
	
	
	
	
	
	
	
	
	
	