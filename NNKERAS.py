# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 23:04:57 2020
@author: Yanes Pérez, Nicolás
@tuthor: Gonzalez Calvo, Daniel 
@version: v0.0
---------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------

"""

# =============================================================================
# LIBRERIAS
# =============================================================================

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# TRATAMIENTO DE LOS DATOS
# =============================================================================

filename = 'CCdata.csv'
dataset = pd.read_csv(filename)

array = dataset.values

scaler = StandardScaler()
scaler.fit(array)
scaled = scaler.transform(array)

X = np.delete(scaled, [4], axis=1)
Y = array[:,4]

validation_size = 0.20
seed = 7

Xtrain, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

# =============================================================================
# RED NEURONAL KERAS
# =============================================================================

# Creamos nuestra red
model = Sequential([
    Dense(125, input_shape=(4,)),
    Activation('selu'),
    Dense(100),
    Activation('relu'),
    Dense(80),
    Activation('relu'),
    Dense(70),
    Activation('relu'),
    Dense(50),
    Activation('relu'),
    Dense(40),
    Activation('relu'),
    Dense(30),
    Activation('relu'),
    Dense(20),
    Activation('relu'),
    Dense(1),
    Activation('relu'),
])
model.compile(loss='mean_squared_error', optimizer='nadam')

for i in range(3):
    # Establecemos el número de épocas y el tamaño del batch
    NUM_EPOCHS = 200
    BATCH_SIZE = 30

    # Realizamos el ajuste del modelo
    history = model.fit(Xtrain, Y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.2, verbose=0)

# Realizamos una prediccion con el modelo utilizado
Y_predicted = model.predict(X_validation).flatten()
    
# =============================================================================
# PUNTUACIONES
# =============================================================================
R2=r2_score(Y_validation,Y_predicted)
MAE=mean_absolute_error(Y_validation,Y_predicted)

print("\nEl valor de R2 es: {:.4f}".format(R2))
print("\nEl valor de MAE es: {:.4f}".format(MAE))

# =============================================================================
# REPRESENTACIÓN
# =============================================================================

plt.plot(np.arange(0,200,1),Y_validation[0:200],'-r',np.arange(0,200,1),Y_predicted[0:200])
plt.legend(['Valores reales','Valores Predichos'])
plt.ylabel('Energía del CC (MW)')
plt.xlabel('Nº Datos')


