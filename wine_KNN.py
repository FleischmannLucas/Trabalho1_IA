# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:47:14 2020

@author: Gustavo
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import numpy as np
from sklearn.model_selection import train_test_split   #nos permite dividir a entrada em grupos
from sklearn.metrics import accuracy_score 
from sklearn.datasets import load_wine


#Carrega o wine dataset em wine 
wine = load_wine()

classe1=wine.data[:58,:]
classe1y=wine.target[:58]
classe1train, classe1test, classe1ytrain, classe1ytest=train_test_split(classe1, classe1y, random_state=2)

classe2=wine.data[59:129,:]
classe2y=wine.target[59:129]
classe2train, classe2test, classe2ytrain, classe2ytest=train_test_split(classe2, classe2y, random_state=2)

classe3=wine.data[130:177,:]
classe3y=wine.target[130:177]
classe3train, classe3test, classe3ytrain, classe3ytest=train_test_split(classe3, classe3y, random_state=2)

Xtest = np.concatenate((classe1test, classe2test, classe3test), axis=0)
Xtrain = np.concatenate((classe1train, classe2train, classe3train), axis=0)
Ytest= np.concatenate((classe1ytest, classe2ytest, classe3ytest), axis=0)
Ytrain= np.concatenate((classe1ytrain, classe2ytrain, classe3ytrain), axis=0)


#Implementa o Algoritmo KNN
neigh = KNeighborsClassifier(n_neighbors=7,weights="uniform")
neigh.fit(Xtrain, Ytrain)

#Prevendo novos valores
Y = neigh.predict(Xtest)


print( accuracy_score( Y, Ytest))



#acuracia para grupos de treino aleatorios e variando o numero de vizinhos em 3-5-7, realizando a media de 5 tentativas.

zn7 = (0.7555555555555555 + 0.7111111111111111  + 0.6222222222222222 + 0.6444444444444445 + 0.7111111111111111)/5 
    # 0.6888888888888889
zn5 = (0.7111111111111111 + 0.7111111111111111 + 0.6222222222222222 + 0.7333333333333333 +0.6888888888888889)/5 
    # 0.6933333333333334
zn3 = (0.7333333333333333 + 0.6666666666666666 +0.7555555555555555 + 0.5777777777777777 +0.6222222222222222 )/5
    # 0.6711111111111111

    

#acuracia para grupos de treino fixos e variando o numero de vizinhos de 3-21 impares.

n21 = 0.7111111111111111
n19 = 0.7333333333333333
n17 = 0.7111111111111111
n15 = 0.7333333333333333
n13 = 0.6666666666666666
n11 = 0.6444444444444445
n9 =  0.6444444444444445
n7 =  0.6666666666666666
n5 =  0.5777777777777777
n3 =  0.5777777777777777



print(classification_report(Y, Ytest))

print( accuracy_score( neigh.predict(classe1test), classe1ytest))
print( accuracy_score( neigh.predict(classe2test), classe2ytest))
print( accuracy_score( neigh.predict(classe3test), classe3ytest))
   

# #print(neigh.predict([[5.9, 3. , 5.1, 1.8]]))







