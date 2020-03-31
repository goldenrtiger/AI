# ---------------------------------------------------------------------
# ---------------------------------------
# 
# python: 3.7
# tensorflow: 2.1.0
# date: 25/03/2020
# author: Xu Jing
# email: xj.yixing@hotmail.com
# 
# ---------------------------------------
# ----------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

 # Generate 200 random points using numpy
 # linspace generates 200 points evenly distributed from -0.5 to 0.5
 # [:,np.newaxis] is to transform one-dimensional data into two-dimensional data
x_data = (np.linspace(-0.5, 0.5, 200)[:, np.newaxis])

 # Generate noise interference point noise
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise # simulation y =x**2

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
                                                x_data, 
                                                y_data, 
                                                random_state = 42,
                                                train_size = 0.8,
                                                test_size = 0.2    
                                            )

layer_0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layer_0])
model.compile(loss = 'mean_squared_error',
            optimizer = tf.keras.optimizers.Adam(0.001)
            )

trained_model = model.fit(
                    x_train, 
                    y_train, 
                    epochs = 1000, 
                    verbose = False
                )
import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(trained_model.history['loss'])
plt.show()

y_pred = model.predict(x_test)

plt.figure()
plt.scatter(x_train, y_train)
plt.plot(x_test, y_pred, 'r-', lw=5)
plt.show()

from sklearn.metrics import r2_score
print(f"r2_score:{r2_score(y_test, y_pred)}")

l_0 = tf.keras.layers.Dense(units=4, input_shape=[1], activation='tanh')
l_1 = tf.keras.layers.Dense(units=10, activation='tanh')
l_2 = tf.keras.layers.Dense(units=1, activation='linear')

model = tf.keras.Sequential([l_0, l_1, l_2])
model.compile(loss='mean_squared_error', optimizer= tf.keras.optimizers.Adam(0.001))
trained_model = model.fit(x_train, y_train, epochs=2000, verbose=False)

plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(trained_model.history['loss'])
plt.show()

y_pred = model.predict(x_test)
print(r2_score(y_test, y_pred))

plt.figure()
plt.scatter(x_train, y_train)
plt.scatter(x_test, y_pred)
plt.plot(x_test, y_test, 'b+', lw=5)
plt.show()









