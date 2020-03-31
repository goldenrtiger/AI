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
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)
# tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt

mar_budget    = np.array([60, 80,  100  , 30, 50, 20, 90,  10],  dtype=float)
subs_gained = np.array([160, 200, 240, 100, 140, 80, 220, 60],  dtype=float)

for i,c in enumerate(mar_budget):
  print("{} Market budget = {} new subscribers gained".format(c, subs_gained[i]))

plt.scatter(mar_budget, subs_gained)
plt.xlim(0,120)
plt.ylim(0,260)
plt.xlabel('Marketing Budget(in thousand of Dollars)')
plt.ylabel('Subscribers Gained(in thousand)')
plt.show()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(mar_budget,subs_gained,random_state=42,
                                               train_size=0.8, test_size=0.2 )

print('Training_features:',X_train)
print('Training_label:' , y_train)
print('\n')
print('Testing_features:', X_test)
print('Testing_labels:',y_test)

layer_0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layer_0])

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))

trained_model = model.fit(X_train, y_train, epochs=1000, verbose=False)
print("Finished training the model")

import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(trained_model.history['loss'])
plt.show()

print(model.predict([80.0]))
# checking for all the test _values
y_pred = model.predict(X_test)
print('Actual Values\tPredicted Values')
print(y_test,'   ',y_pred.reshape(1,-1))

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

print("These are the layer variables: {}".format(layer_0.get_weights()))

l_0 = tf.keras.layers.Dense(units=4, input_shape=[1])
l_1 = tf.keras.layers.Dense(units=5)
l_2 = tf.keras.layers.Dense(units=1) 
model = tf.keras.Sequential([l_0, l_1, l_2])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
model.fit(X_train,y_train, epochs=2000,verbose=False)
print('\n Finished training Model')

print(model.predict([80]))
y_pred=model.predict(X_test)
print(r2_score(y_test,y_pred))

## TODO: print the wieght of each layer and think why are they so different from (2,40). 











