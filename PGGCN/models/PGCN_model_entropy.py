from . import layers_update_mobley as layers
import importlib
import keras.backend as K
import numpy as np
import tensorflow as tf
# import tensorflow_addons as tfa
importlib.reload(layers)


class PGGCNModel(tf.keras.Model):
    def __init__(self, num_atom_features=36, r_out_channel=20, c_out_channel=128):
        super().__init__()
        self.ruleGraphConvLayer = layers.RuleGraphConvLayer(r_out_channel, num_atom_features, 0)
        self.ruleGraphConvLayer.combination_rules = []
        self.conv = layers.ConvLayer(c_out_channel)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', name='dense1')
        self.dense5 = tf.keras.layers.Dense(16, name='relu')
        self.dense6 = tf.keras.layers.Dense(1, name='dense6')
        self.dense7 = tf.keras.layers.Dense(1, name='dense7',
                                            kernel_initializer=tf.keras.initializers.Constant([-.2, -1, 1, 1]),
                                            bias_initializer=tf.keras.initializers.Zeros())
        self.all_layer_1_weights = []

    def addRule(self, rule, start_index, end_index=None):
        self.ruleGraphConvLayer.addRule(rule, start_index, end_index)

    def set_input_shapes(self, i_s):
        self.i_s = i_s

    def call(self, inputs):
        physics_info = inputs[:, 0, 38:]
        x_a = []
        for i in range(len(self.i_s)):
            x_a.append(inputs[i][:self.i_s[i], :38])
        x = self.ruleGraphConvLayer(x_a)
        self.all_layer_1_weights.append(self.ruleGraphConvLayer.w_s)
        x = self.conv(x)
        x = self.dense1(x)
        x = self.dense5(x)
        model_var = self.dense6(x)
        merged = tf.concat([model_var, physics_info], axis=1)
        out = self.dense7(merged)
        return out

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred[0] - y_true))) + K.abs(1 / K.mean(.2 + y_pred[1]))


def pure_rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def get_trained_model(X, y, epochs = 1, max_num_atoms = 2000, n_features = 41):
    m = PGGCNModel()
    m.addRule("sum", 0, 31)
    m.addRule("multiply", 31, 33)
    m.addRule("distance", 33, 36)

    m.compile(loss=pure_rmse, optimizer='adam')

    input_shapes = []
    for i in range(len(X)):
        input_shapes.append(np.array(X[i]).shape[0])
    m.set_input_shapes(input_shapes)
    for i in range(len(X)):
        if X[i].shape[0] < max_num_atoms:
            new_list = np.zeros([max_num_atoms - X[i].shape[0], n_features])
            X[i] = np.concatenate([X[i], new_list], 0)
    X_train = np.array(X)
    y_train = np.array(y)
    hist = m.fit(X_train, y_train, epochs=epochs, batch_size=len(X_train))
    return hist, m

def test_model(X_test, y_test, m: PGGCNModel, max_num_atoms = 2000, n_features = 41):
    input_shapes = []
    for i in range(len(X_test)):
        input_shapes.append(np.array(X_test[i]).shape[0])
    m.set_input_shapes(input_shapes)
    for i in range(len(X_test)):
        if X_test[i].shape[0] < max_num_atoms:
            new_list = np.zeros([max_num_atoms - X_test[i].shape[0], n_features])
            X_test[i] = np.concatenate([X_test[i], new_list], 0)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    eval = m.evaluate(X_test, y_test)
    return eval