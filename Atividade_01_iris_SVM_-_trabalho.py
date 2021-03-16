# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 21:44:40 2021

@author: Lucas Gava
"""



from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split   #nos permite dividir a entrada em grupos
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report

#Carrega o iris dataset em iris 

iris = load_iris() 


#Divisao dos grupos de treino e teste com quantidades iguais de cada grupo
cetosa= iris.data[:50,:]
cetosay= iris.target[:50]
cet=np.array(cetosa)
cety=np.array(cetosay)
cettrain, cettest, cetytrain, cetytest=train_test_split(cet, cety, random_state=0)

versicolor= iris.data[50:100,:]
versicolory= iris.target[50:100]
ver=np.array(versicolor)
very=np.array(versicolory)
vertrain, vertest, verytrain, verytest=train_test_split(ver, very, random_state=0)

virginica= iris.data[100:150,:]
virginicay= iris.target[100:150]
virg=np.array(virginica)
virgy=np.array(virginicay)
virgtrain, virgtest, virgytrain, virgytest=train_test_split(virg, virgy, random_state=0)

Xtest = np.concatenate((cettest, vertest, virgtest), axis=0)
Xtrain = np.concatenate((cettrain, vertrain, virgtrain), axis=0)
Ytest = np.concatenate((cetytest, verytest, virgytest), axis=0)
Ytrain = np.concatenate((cetytrain, verytrain, virgytrain), axis=0)


#iniciação do classificador SVM
clf = svm.SVC(C=1.0)
clf.fit(Xtrain,Ytrain)


print(clf.predict(Xtest)) #imprime os resultados

Z = clf.predict(Xtest)

print(accuracy_score(Z, Ytest))     #imprime acuracia total

print(classification_report(Z, Ytest))   #imprime tabela de acuracia


print(accuracy_score(clf.predict(cettest), cetytest))  #calcula a acuracia para as cetosas
print(accuracy_score(clf.predict(vertest), verytest))  #calcula a acuracia para as versicolors
print(accuracy_score(clf.predict(virgtest), virgytest))  #calcula a acuracia para as virginicas


