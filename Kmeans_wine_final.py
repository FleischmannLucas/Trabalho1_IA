# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 20:00:14 2021

@author: Lucas Gava
"""

from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split   #nos permite dividir a entrada em grupos
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import classification_report

wine=load_wine()

tipo1=wine.data[:58,:]
tipo1y=wine.target[:58]
tipo11, tipo12, tipoy11, tipoy12=train_test_split(tipo1, tipo1y, random_state=0)

tipo2=wine.data[59:129,:]
tipo2y=wine.target[59:129]
tipo21, tipo22, tipoy21, tipoy22=train_test_split(tipo2, tipo2y, random_state=0)

tipo3=wine.data[130:177,:]
tipo3y=wine.target[130:177]
tipo31, tipo32, tipoy31, tipoy32=train_test_split(tipo3, tipo3y, random_state=0)

parte1_test = np.concatenate((tipo12, tipo22, tipo32), axis=0)
parte2_train = np.concatenate((tipo11, tipo21, tipo31), axis=0)
parte1y_test= np.concatenate((tipoy12, tipoy22, tipoy32), axis=0)
parte2y_train= np.concatenate((tipoy11, tipoy21, tipoy31), axis=0)

#iris.target
#Implementa o Algoritmo K-means
kmeans = KMeans(n_clusters=3, random_state=78).fit(parte2_train)
kmeans.labels_
kmeans.cluster_centers_


# transforma os labels de saida para os mesmos dos dados de treino

Z=np.zeros((45,))
x=kmeans.predict(parte1_test)
for i in range(0,45,1):
    # print (i)
    if (x[i]==1):
        Z[i] = 0
    if (x[i]==0):
        Z[i] = 1
    if (x[i]==2):
        Z[i] = 2



print(accuracy_score(Z, parte1y_test)) #calcula a acertividade geral todos os grupos juntos 

print(classification_report(Z, parte1y_test))
    
    

