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
import tensorflow               as tf
import numpy                    as np
import matplotlib.pylab         as plt
from sklearn.metrics            import r2_score
from sklearn.model_selection    import train_test_split

def single_layer_test():
    print(">>>>>>>>>>>>> start single_layer_test -------------------------")
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
    print(f"r2_score:{r2_score(y_test, y_pred)}")

def multiple_layers_test():
    print(">>>>>>>>>>>>> start multiple_layers_test -------------------------")
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

# ---------------------------------------- another test ----------------------------------------------
def multiple_layers_complex_test():
    print(">>>>>>>>>>>>> start multiple_layers_complex_test -------------------------")
    x_data = np.abs(np.random.rand(4, 200).astype(np.float32) )
    w = np.abs(np.array([1., 1., 1., 1.]))

    [s, fm, Zn, Vr] = x_data
    [ a, b, c, d ] = w

    # Generate noise interference point noise
    noise = np.random.normal(0, 0.02, 200)
    y_data = s.T* tf.math.pow(fm.T, a) * tf.math.pow(Zn.T, b) * c / tf.math.pow(Vr.T, d) + noise # simulation y =x**2
    y_data = y_data.numpy()

    x_train, x_test, y_train, y_test = train_test_split(
                                                    x_data.T, 
                                                    y_data, 
                                                    random_state = 42,
                                                    train_size = 0.75,
                                                    test_size = 0.25   
                                            )

    l_0 = tf.keras.layers.Dense(units=8, input_shape=[4], activation='tanh')
    l_1 = tf.keras.layers.Dense(units=20, activation='tanh')
    l_2 = tf.keras.layers.Dense(units=20, activation='tanh')
    l_3 = tf.keras.layers.Dense(units=1, activation='linear')

    model = tf.keras.Sequential([l_0, l_1, l_2, l_3])
    model.compile(loss='mean_squared_error', optimizer= tf.keras.optimizers.SGD(0.001))
    trained_model = model.fit(x_train, y_train, epochs=10000, verbose=False)

    plt.xlabel('Epoch Number')
    plt.ylabel("Loss Magnitude")
    plt.plot(trained_model.history['loss'])
    plt.show()

    y_pred = model.predict(x_test)
    print(r2_score(y_test, y_pred))
    l = len(y_pred)

    x = np.linspace(1, l, l)

    plt.scatter(x, y_test, color='b', label='test')
    plt.scatter(x, y_pred, color='r', label='predict')
    plt.legend()
    plt.show()

def test(index):
    if index == 0:
        single_layer_test()
    elif index == 1:
        multiple_layers_test()
    elif index == 2:
        multiple_layers_complex_test()
        
test(2)
