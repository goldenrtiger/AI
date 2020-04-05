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

# --------------------------- tensor, Variable, constant ---------------------------------------
# tf.constant
# tf.convert_to_tensor
# tf.Variable
# shape
# .numpy()
# tf.transpose
# assign
# tensor iterate => map_fn
# tf.stack axis = 0
# matrix
# one-like
# tf.cast
# tensor dynamic shape
#
# -------------------------------------------------------------------------------------------

import tensorflow as tf

# the below script is in eager mode.
a = tf.constant([1.0, 1.0])
print(a)
# >>>out:
#     tf.Tensor([1. 1.], shape=(2,), dtype=float32)
    
tf.print(a)
# >>>out:
#     [1 1]

import numpy as np

def my_func(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return tf.matmul(arg, arg) + arg

# The following calls are equivalent.
value_1 = my_func(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
value_2 = my_func([[1.0, 2.0], [3.0, 4.0]])
value_3 = my_func(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

tf.convert_to_tensor([1,2]) 
# >>>out:
#     <tf.Tensor: id=0, shape=(2,), dtype=int32, numpy=array([1, 2])>
        
a = tf.Variable([[1,2,3],[2,3,4]],dtype=tf.float32) 
a.shape
# >>>out:
# TensorShape([2, 3])
a[0].shape
# >>>out:
# TensorShape([3]) 
tf.shape(a[0]).numpy()
# array([3])
# >>>out:
tf.shape(a[0]).numpy()[0]
# >>>out:
# 3
tf.shape(a[0])[0]
# >>>out:
# <tf.Tensor: shape=(), dtype=int32, numpy=3>
a[0].shape.numpy()
# >>>out:
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# AttributeError: 'TensorShape' object has no attribute 'numpy'

a 
# >>>out:
# <tf.Variable 'Variable:0' shape=(2, 3) dtype=float32, numpy=
# array([[1., 2., 3.],
#        [2., 3., 4.]], dtype=float32)>
tf.transpose(a) 
# >>>out:
# <tf.Tensor: shape=(3, 2), dtype=float32, numpy=
# array([[1., 2.],
#        [2., 3.],
#        [3., 4.]], dtype=float32)>

a + a
# >>>out:
# <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
# array([[2., 4., 6.],
#        [4., 6., 8.]], dtype=float32)>

a[0].assign(tf.Variable([1.1,1,1])) 
# >>>out:
# <tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32, numpy=
# array([[1.1, 1. , 1. ],
#        [2. , 3. , 4. ]], dtype=float32)>

# ------------------------- iterate tensor ---------------------------------------------
import numpy as np
elems = np.array([1, 2, 3, 4, 5, 6])
squares = tf.map_fn(lambda x: x * x, elems) 
squares
# >>>out:
# <tf.Tensor: shape=(6,), dtype=int32, numpy=array([ 1,  4,  9, 16, 25, 36])>

# -------------------------- stack --------------------------------------------------
tf.stack([a,a,a],axis=0)
# >>>out:
# <tf.Tensor: shape=(3, 2, 3), dtype=float32, numpy=
# array([[[1.1, 1. , 1. ],
#         [2. , 3. , 4. ]],

#        [[1.1, 1. , 1. ],
#         [2. , 3. , 4. ]],

#        [[1.1, 1. , 1. ],
#         [2. , 3. , 4. ]]], dtype=float32)>
tf.Variable([a,a,a])
# >>>out:
# <tf.Variable 'Variable:0' shape=(3, 2, 3) dtype=float32, numpy=
# array([[[1.1, 1. , 1. ],
#         [2. , 3. , 4. ]],

#        [[1.1, 1. , 1. ],
#         [2. , 3. , 4. ]],

#        [[1.1, 1. , 1. ],
#         [2. , 3. , 4. ]]], dtype=float32)>

c
# >>>out:
# <tf.Tensor: shape=(3, 3), dtype=int32, numpy=
# array([[1, 1, 1],
#        [1, 1, 1],
#        [1, 1, 1]])>
c[0] 
# >>>out:
# <tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 1, 1])>
tf.Variable(c[0]) 
# >>>out:
# <tf.Variable 'Variable:0' shape=(3,) dtype=int32, numpy=array([1, 1, 1])>
tf.Variable([c[0]]) 
# >>>out:
# <tf.Variable 'Variable:0' shape=(1, 3) dtype=int32, numpy=array([[1, 1, 1]])>
d = tf.Variable([c[0]]) 
d
# >>>out:
# <tf.Variable 'Variable:0' shape=(1, 3) dtype=int32, numpy=array([[1, 1, 1]])>
d[0] 
# >>>out:
# <tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 1, 1])>
d[0][0] 
# >>>out:
# <tf.Tensor: shape=(), dtype=int32, numpy=1>
    
a
# >>>out:
# <tf.Variable 'Variable:0' shape=(3,) dtype=int32, numpy=array([1, 1, 1])>
b
# >>>out:
# <tf.Variable 'Variable:0' shape=(3,) dtype=int32, numpy=array([1, 1, 1])>
c
# >>>out:
# <tf.Variable 'Variable:0' shape=(2, 3) dtype=int32, numpy=
# array([[1, 1, 1],
#        [1, 1, 1]])>
d = tf.Variable([a])
d
# >>>out:
# <tf.Variable 'Variable:0' shape=(1, 3) dtype=int32, numpy=array([[1, 1, 1]])>
d[0]
# >>>out:
# <tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 1, 1])>
tf.Variable([d[0],d[0]])
# >>>out:
# <tf.Variable 'Variable:0' shape=(2, 3) dtype=int32, numpy=
# array([[1, 1, 1],
#        [1, 1, 1]])>

a = tf.Variable([[1,2,3]]) 
b = tf.Variable([[1,2,3]])
tf.stack([a,b], axis=0)
# >>>out:
# <tf.Tensor: shape=(2, 1, 3), dtype=int32, numpy=
# array([[[1, 2, 3]],

#        [[1, 2, 3]]])>
tf.stack([a,b], axis=1)
# >>>out:
# <tf.Tensor: shape=(1, 2, 3), dtype=int32, numpy=
# array([[[1, 2, 3],
#         [1, 2, 3]]])>

b
# >>>out:
# <tf.Variable 'Variable:0' shape=(3,) dtype=int32, numpy=array([1, 2, 3])>
tf.stack([b,b], axis = 1) 
# >>>out:
# <tf.Tensor: shape=(3, 2), dtype=int32, numpy=
# array([[1, 1],
#        [2, 2],
#        [3, 3]])>


# matrix
tf.constant(np.array([1,2], dtype = tf.float32))

# ------------ ones_like ----------------------------
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
tf.ones_like(tensor)  # [[1, 1, 1], [1, 1, 1]]

# >>>out:
# <tf.Tensor: shape=(2, 3), dtype=int32, numpy=
# array([[1, 1, 1],
#        [1, 1, 1]])>

# ----------------- tf.cast --------------------------------------------
x = tf.constant([1.8, 2.2], dtype=tf.float32)
x = tf.dtypes.cast(x, tf.int32)  # [1, 2], dtype=tf.int32

# ----------------- dynamic shape ----------------------------------------
a = tf.Variable(2, shape=tf.TensorShape(None)) 
a
# >>>out:
# <tf.Variable 'Variable:0' shape=<unknown> dtype=int32, numpy=2>
a.assign([1,2,3]) 
# >>>out:
# <tf.Variable 'UnreadVariable' shape=<unknown> dtype=int32, numpy=array([1, 2, 3])>
a.assign([1,2,3,4]) 
# >>>out:
# <tf.Variable 'UnreadVariable' shape=<unknown> dtype=int32, numpy=array([1, 2, 3, 4])>

tf.reduce_sum([aa,aa]) 
# >>>out:
# <tf.Tensor: shape=(), dtype=float32, numpy=6.0>
tf.reduce_sum([b,b])  
# >>>out: 
# <tf.Tensor: shape=(), dtype=float32, numpy=60.0>








