# ---------------------------------------------------------------------
# ---------------------------------------
# 
# python: 3.7
# tensorflow: 2.1.0
# date: 31/03/2020
# author: Xu Jing
# email: xj.yixing@hotmail.com
# 
# ---------------------------------------
# ----------------------------------------------------------------------
import tensorflow as tf

var1 = tf.Variable([1.0], name='var1',trainable=True, dtype=tf.float32) 
var2 = tf.Variable([2.0], name='var2',trainable=True, dtype=tf.float32) 
opt = tf.keras.optimizers.SGD(learning_rate=0.1)

print(">>>>>>>>>>>> start!! gradient->apply_gradients----------------------")

with tf.GradientTape(watch_accessed_variables=True) as g:   
    y = 3 * var1 * var1 + 2 * var2 * var2

grads = g.gradient(y, [var1, var2])
opt.apply_gradients(zip(grads, [var1, var2]))
y = 3 * var1 * var1 + 2 * var2 * var2

print(f"<<< grads:{grads} \n "
        f"<<< y:{y} \n "
        f"<<< var1: {var1} \n"
        f"<<< var2: {var2} \n")

print(">>>>>>>>>>>> start!! minimize ----------------------")
var1 = tf.Variable([1.0], name='var1',trainable=True, dtype=tf.float32) 
var2 = tf.Variable([2.0], name='var2',trainable=True, dtype=tf.float32) 

opt = tf.keras.optimizers.SGD(learning_rate=0.1)

opt_op=opt.minimize(lambda: 3*var1*var1+2*var2*var2, var_list=[var1,var2]) 

y = 3*var1*var1+2*var2*var2
print(f"<<< var1: {var1} \n"
        f"<<< y:{y} \n "
        f"<<< var2: {var2} \n")    


