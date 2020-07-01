# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 00:43:00 2020
@author: Yanes Pérez, Nicolás
@tuthor: Gonzalez Calvo, Daniel 
@version: v0.0
---------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------

"""

# =============================================================================
# LIBRERIAS
# =============================================================================

import lightgbm as lgb 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score 
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

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

Xtrain, X_validation, Ytrain, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)


# =============================================================================
# MODELO EMSEMBLE LIGHT GRADIENT BOOSTING
# =============================================================================

# MODELO INICIAL
lgbm = lgb.LGBMRegressor(
    boosting_type="gbdt",
    objective='regression', 
    random_state=10, 
    n_estimators=320,
    num_leaves=20,
    max_depth=8,
    feature_fraction=0.9,  
    bagging_fraction=0.8, 
    bagging_freq=15, 
    learning_rate=0.01)
    
    
# OBTENEMOS LOS MEJORES PARAMETROS PARA EL MODELO
score_func = make_scorer(r2_score, greater_is_better=True)
params_opt = {'n_estimators':range(200, 600, 100), 'num_leaves':range(20,60,10)}
gridSearchCV = GridSearchCV(estimator = lgbm, param_grid = params_opt, scoring=score_func, cv=5)
gridSearchCV.fit(Xtrain,Ytrain)
params = gridSearchCV.best_params_


lgbm = lgb.LGBMRegressor(
    boosting_type="gbdt",
    objective='regression', 
    random_state=10, 
    n_estimators=params['n_estimators'],
    num_leaves=params['num_leaves'],
    max_depth=8,
    feature_fraction=0.9,  
    bagging_fraction=0.8, 
    bagging_freq=15, 
    learning_rate=0.01)

# =============================================================================
# ENTRENAMIENTO DEL MODELO YA OPTIMIZADO
# =============================================================================

# ENTRENAMOS EL MODELO
lgbm.fit(Xtrain, Ytrain,
          eval_set=[(Xtrain, Ytrain)],
          eval_metric='l2',
          early_stopping_rounds=5,
          verbose=False)

# =============================================================================
# PREDICCIÓN DEL MODELO
# =============================================================================

y_pred = lgbm.predict(X_validation, num_iteration=lgbm.best_iteration_).flatten()

# =============================================================================
# PUNTUACIONES
# =============================================================================

R2=r2_score(Y_validation,y_pred)
MAE=mean_absolute_error(Y_validation,y_pred)

print("\nEl valor de R2 es: {:.4f}".format(R2))
print("\nEl valor de MAE es: {:.4f}".format(MAE))

# =============================================================================
# REPRESENTACIÓN
# =============================================================================

plt.plot(np.arange(0,200,1),Y_validation[0:200],'-r',np.arange(0,200,1),y_pred[0:200])
plt.legend(['Valores reales','Valores Predichos'])
plt.ylabel('Energía del CC (MW)')
plt.xlabel('Nº Datos')
