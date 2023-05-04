# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 13:50:58 2023

@author: Cristian
"""
#Red neuronal buena
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

 # color = np.array([7,247,26, 120,217,(236),254,128],dtype=float)#cada vex que hay parentesis es unan medida nueva
# temp = np.array([18.6, 44.8,20.6,30.1, 37.8,38.5,43.1,28.6], dtype=float)
color = np.array([
223.97979797979798,
228.06796116504853,
227.16822429906543,
218.6705882352941,
218.51612903225808,
224.30275229357798,
229.03378378378378, #entre_16
219.57251908396947,#entre_17
228.67605633802816,#entre_18
230.1418918918919,#entre_19
229.82539682539684,#entre_20
225.05633802816902,#entre 21
223.3728813559322,#entre23
218.8,
214.72815533980582,
215.49473684210525,
228.49612403100775,
228.93965517241378,
231.1851851851852,
229.6640625,#entre_30
227.1315789473684,
228.56603773584905
],dtype=float)#cada vex que hay parentesis es unan medida nueva

color=color/255

temp = np.array([
35.6,
36.2,
36.3,
36.0,
35.6,
36.1,
36.1,
36.2,
35.9,
35.9,
35.9,#entre20
36.3,#entre 21
36.3,#entre 23
35.6,
35.6,
35.6,
36.1,
36.1,
36.1,
36.1,
36.1,
36.1
], dtype=float)

amb = np.array([
20.3,
20.3,
20.1,
20.3,
20.1,
21.1,
24.8,
25.7,
24.6,
24.7,
24.7,#entre 20
23.1,#entre 21
23.2,#entre 23
23.3,
23.3,
23.3,
23.3,
23.3,
23.3,
22.1,
22.1,
22.1
], dtype=float)
amb_media = np.mean(amb)
amb_desvi = np.std(amb)
amb_normalizado= (amb - amb_media ) / amb_desvi 

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=2))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compilación del modelo
model.compile(optimizer='adam', loss='mean_squared_error')
inputs = np.concatenate([color.reshape(-1, 1), amb_normalizado.reshape(-1, 1)], axis=1)
# Entrenamiento del modelo
model.fit(x=inputs, y=temp, epochs=583, batch_size=32, validation_split=0.2)

# Evaluación del modelo
# test_loss = model.evaluate(x=[normalized_test_pixels, normalized_test_ambient_temperatures], y=test_temp)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import triangulation as mtri
# plt.figure()
# plt.xlabel("# Epoca")
# plt.ylabel("Magnitud de pérdida")
# plt.plot(model.fit.history["loss"])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(color, temp, amb, c=color)
ax.set_xlabel('Color')
ax.set_ylabel('Temperatura')
ax.set_zlabel('Ambiente')
plt.show()
plt.figure()
plt.ylabel("color")
plt.xlabel("temperatura")

plt.plot(  temp, color,'ro')

plt.figure()
plt.xlabel("color")
plt.ylabel("ambiente")

plt.plot( color, amb, 'ro')

plt.figure()
plt.ylabel("ambiente")
plt.xlabel("temperatura")

plt.plot(  temp, amb,'ro')
print("Hagamos una predicción!")
ambiente_test=23.8
ambiente_test=(ambiente_test- amb_media ) / amb_desvi 

# resultado = model.predict([[(226.31147540983608/255), ambiente_test]])
# print("El resultado es " + str(resultado) + "ºC")
# model.save('mi_modelo2.h5')