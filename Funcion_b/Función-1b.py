import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam

from matplotlib import pyplot as plt
import numpy as np


# el dominio de la funcion es [-1,1]
x = tf.linspace(-1,1,500)
y = 1+(2*x)+(4*(x**3)) #Funcion a reproducir

model = Sequential()
model.add(Dense(20, activation='relu',input_shape=(1,)))
model.add(Dense(20, activation ='relu'))
model.add(Dense(1, activation = 'linear'))

model.summary()

model.compile(optimizer=RMSprop(),loss='mse')

history = model.fit(x,y,epochs=500,batch_size=10)

f = model.predict(x)
plt.plot(x,f,label='$f(x)$ de la red',color = 'k')
plt.plot(x,y,label='$f(x)$',color = 'r',ls='--')
plt.grid()
plt.legend()
plt.show()
model.save('red1b_T3.h5')

modelo_cargado = tf.keras.models.load_model("red1b_T3.h5") 