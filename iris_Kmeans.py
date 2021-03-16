# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 19:22:41 2021

@author: Gustavo
"""

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split   #nos permite dividir a entrada em grupos
from sklearn.metrics import accuracy_score
import numpy as np

#Carrega o iris dataset em iris 

iris = load_iris() 
# iristreino=iris.data[25:125,:]

X, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0) #divide a poha toda


# Xtreino,Xtest,Ytreino,Ytest = train_test_split(data,features)


# Como o anterior pega 25% dos dados aleatoriamente, e a maiores resultados corretos
# para as cetosas, decidimos redivir afim de pegar uma parcela igual de plantas
# de cada categoria, assim garantimos uma maior uniformidade nos resultados


cetosa= iris.data[:50,:]
cetosay= iris.target[:50]
cet=np.array(cetosa)
cety=np.array(cetosay)
cettrain, cettest, cetytrain, cetytest=train_test_split(cet, cety, random_state=5)

versicolor= iris.data[50:100,:]
versicolory= iris.target[50:100]
ver=np.array(versicolor)
very=np.array(versicolory)
vertrain, vertest, verytrain, verytest=train_test_split(ver, very, random_state=5)

virginica= iris.data[100:150,:]
virginicay= iris.target[100:150]
virg=np.array(virginica)
virgy=np.array(virginicay)
virgtrain, virgtest, virgytrain, virgytest=train_test_split(virg, virgy, random_state=5)

X_test = np.concatenate((cettest, vertest, virgtest), axis=0)
X_train = np.concatenate((cettrain, vertrain, virgtrain), axis=0)
Y_test = np.concatenate((cetytest, verytest, virgytest), axis=0)
Y_train = np.concatenate((cetytrain, verytrain, virgytrain), axis=0)

#iris.target
#Implementa o Algoritmo K-means
kmeans = KMeans(n_clusters=3, random_state=11).fit(X_train)
kmeans.labels_
kmeans.cluster_centers_
#Prevendo novos valores
#kmeans.predict([[5, 6, 7, 8], [4, 3, 4, 4]])
accuracy_score(kmeans.predict(X_test), Y_test)  #calcula a acertividade geral todos os grupos juntos 

accuracy_score(kmeans.predict(cettest), cetytest)  #calcula a acuracia para as cetosas
accuracy_score(kmeans.predict(vertest), verytest)  #calcula a acuracia para as versicolors
accuracy_score(kmeans.predict(virgtest), virgytest)  #calcula a acuracia para as virginicas