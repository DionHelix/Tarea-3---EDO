import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
from matplotlib import pyplot as plt
import numpy as np

class ODEsolver(Sequential):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        
    @property
    def metrics(self):
        return[self.loss_tracker]
    
    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        x = tf.random.uniform((batch_size, 1), minval=-5, maxval=5)
        x_0 = tf.zeros((batch_size, 1))
        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                with tf.GradientTape(persistent=True) as tape3:
                    tape3.watch(x)
                    tape3.watch(x_0)
                    y_pred = self(x, training=True)
                    y_0 = self(x_0, training=True)
                dy = tape3.gradient(y_pred, x)
                dy_0 = tape3.gradient(y_0, x_0)
            dy2 = tape2.gradient(dy, x)    
            print('Forma de x: ', x)
            print('Forma de x_0: ', x_0)
            print('Forma de y_pred: ', y_pred)
            print('Forma de y_0: ', y_0)
            print('Forma de dy: ', dy)
            print('Forma de dy_0: ', dy_0)
            eq = dy2 + y_pred
            ic = y_0 - 1
            ic_prima = dy_0 + 0.5
            loss = keras.losses.mean_squared_error(0., eq) + keras.losses.mean_squared_error(0., ic) + keras.losses.mean_squared_error(0., ic_prima)
            
        grads =  tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

model = ODEsolver()

model.add(Dense(10, activation='tanh', input_shape=(1,))) 
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(1, activation='linear'))

model.summary()

model.compile(optimizer=RMSprop(), metrics=['loss'])
x=tf.linspace(-5, 5, 100)
history = model.fit(x, epochs=50, verbose = 1)

x_testv = tf.linspace(-5, 5, 100)
a=model.predict(x_testv)
plt.plot(x_testv, a)
plt.plot(x_testv, np.cos(x)-0.5*np.sin(x))
plt.show()