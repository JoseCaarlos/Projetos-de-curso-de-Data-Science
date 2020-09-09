# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 16:36:11 2020

@author: Matheus
"""

import pandas as pd
#ler os dados de entrada e saida
datasetInputs = pd.read_csv("entradas_breast.csv")
datasetOutputs =  pd.read_csv("saidas_breast.csv")

from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(datasetInputs,datasetOutputs, test_size=0.25)

#criando a rede
import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform',input_dim=30))
model.add(Dense(units = 1, activation ='sigmoid'))
model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics =['binary_accuracy'])
model.fit(X_treinamento,y_treinamento,batch_size=10,epochs=100)

y_predicted = model.predict(X_teste)
y_predicted =(y_predicted > 0.5)

from sklearn.metrics import confusion_matrix,accuracy_score
perfomance_output = accuracy_score(y_teste,y_predicted)
matriz = confusion_matrix(y_teste,y_predicted)

result = model.evaluate(X_teste,y_teste)

pesos0_1 = model.layers[0].get_weights()
