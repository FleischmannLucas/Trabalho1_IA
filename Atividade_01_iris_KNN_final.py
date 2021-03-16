# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:47:14 2020

@author: Gustavo
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
import numpy as np
from sklearn.model_selection import train_test_split   #nos permite dividir a entrada em grupos
from sklearn.metrics import accuracy_score 


#Carrega o iris dataset em iris 
iris = load_iris()
X = iris.data
y = iris.target 


#divisao do data set entra grupo de treino e test
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

Xtest = np.concatenate((cettest, vertest, virgtest), axis=0)
Xtrain = np.concatenate((cettrain, vertrain, virgtrain), axis=0)
Ytest = np.concatenate((cetytest, verytest, virgytest), axis=0)
Ytrain = np.concatenate((cetytrain, verytrain, virgytrain), axis=0)



#Implementa o Algoritmo KNN
neigh = KNeighborsClassifier(n_neighbors=5,weights="uniform")
neigh.fit(Xtrain, Ytrain)

#Prevendo novos valores
Y = neigh.predict(Xtest)


print( accuracy_score( Y, Ytest))



#acuracia para grupos de treino aleatorios e variando o numero de vizinhos em 3-5-7, realizando a media de 5 tentativas.

zn7 = (0.8974358974358975 + 1  + 0.9487179487179487 + 0.9743589743589743 + 0.8717948717948718)/5 
zn5 = (0.8974358974358975 + 0.9487179487179487 + 0.9743589743589743 + 0.9487179487179487 +0.9487179487179487)/5
zn3 = (0.9743589743589743 + 0.9487179487179487 +0.9743589743589743 + 0.9743589743589743 +0.9487179487179487 )/5

#acuracia para grupos de treino fixos e variando o numero de vizinhos em 3-5-7.



n7 =  0.9743589743589743
n5 =  0.9487179487179487
n3 =  0.9487179487179487

#acredito que a pouca variacao se deve a pequena amostra de test com apenas 39 entradas



print(classification_report(Y, Ytest))

    
   

#print(neigh.predict([[5.9, 3. , 5.1, 1.8]]))







