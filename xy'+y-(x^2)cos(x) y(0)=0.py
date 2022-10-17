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
        
        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                with tf.GradientTape() as tape3:
                    tape3.watch(x)
                    y_pred = self(x, training=True)    
                dy = tape3.gradient(y_pred, x)
            dy2 = tape2.gradient(dy, x)    
            x_0 = tf.zeros((batch_size), 1)
            y_0 = self(x_0, training=True)
            eq = x*dy + y_pred - tf.math.cos(x)*x**2
            ic = y_0
            loss = keras.losses.mean_squared_error(0., eq) + keras.losses.mean_squared_error(0., ic)
            
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

model.compile(optimizer=Adam(), metrics=['loss'])
x=tf.linspace(-5, 5, 100)
history = model.fit(x, epochs=500, verbose = 1)

x_testv = tf.linspace(-5, 5, 100)
a=model.predict(x_testv)
plt.plot(x_testv, a)
plt.plot(x_testv, -1+np.cos(x)+x*np.sin(x))
plt.show()