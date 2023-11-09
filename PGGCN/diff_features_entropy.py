import pandas as pd
import tensorflow as tf
import numpy as np
from deepchem.feat.graph_features import atom_features as get_atom_features
import pickle
import importlib
import keras.backend as K

import models.layers_update_mobley as layers
importlib.reload(layers)
from models.dcFeaturizer import atom_features as get_atom_features
# %cd ../Notebooks/Entropy
PDBs = pickle.load(open('PDBs_RDKit.pkl', 'rb'))

print(len(PDBs.keys()))
def featurize(molecule, info, exclude=None):
    atom_features = []
    for atom in molecule.GetAtoms():
        new_feature = get_atom_features(atom).tolist()
        position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
        new_feature += [atom.GetMass(), atom.GetAtomicNum()]
        new_feature += [position.x, position.y, position.z]
        for neighbor in atom.GetNeighbors()[:2]:
            neighbor_idx = neighbor.GetIdx()
            new_feature += [neighbor_idx]
        for i in range(2 - len(atom.GetNeighbors())):
            new_feature += [-1]

        if exclude != None:
            for i in exclude:
                new_feature.pop(i)

        atom_features.append(np.concatenate([new_feature, info], 0))
    return np.array(atom_features)


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
        physics_info = inputs[:, 0, 38 - 1:]
        x_a = []
        for i in range(len(self.i_s)):
            x_a.append(inputs[i][:self.i_s[i], :38 - 1])
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

def train(X, y, excelude):
    ex = len(excelude)
    m = PGGCNModel(num_atom_features = 36-ex)
    m.addRule("sum", 0, 31-ex)
    m.addRule("multiply", 31-ex, 33-ex)
    m.addRule("distance", 33-ex, 36-ex)

    m.compile(loss=pure_rmse, optimizer='adam')
    X_train, X_test, y_train, y_test = X[:int(.8*len(X))], X[int(.8*len(X)):], y[:int(.8*len(X))], y[int(.8*len(X)):]

    input_shapes = []
    for i in range(len(X_train)):
        input_shapes.append(np.array(X_train[i]).shape[0])
    m.set_input_shapes(input_shapes)
    for i in range(len(X_train)):
        if X_train[i].shape[0] < 2000:
            new_list = np.zeros([2000 - X_train[i].shape[0], X_train[i].shape[-1]])
            X_train[i] = np.concatenate([X_train[i], new_list], 0)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    hist = m.fit(X_train, y_train, epochs = 25, batch_size=100)

    input_shapes = []
    y_test = y[int(.8*len(X)):]
    for i in range(len(X_test)):
        input_shapes.append(np.array(X_test[i]).shape[0])
    m.set_input_shapes(input_shapes)
    for i in range(len(X_test)):
        if X_test[i].shape[0] < 2000:
            new_list = np.zeros([2000 - X_test[i].shape[0], 40])
            X_test[i] = np.concatenate([X_test[i], new_list], 0)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    return m.evaluate(X_test, y_test), hist


def experiment(excelude):
    PDBs = pickle.load(open('PDBs_RDKit.pkl', 'rb'))
    df = pd.read_csv('T_data.csv')
    info = []
    for pdb in list(PDBs.keys()):
        info.append(df[df['Id'] == pdb][['TS_comp', 'TS_host', 'TS_ligand']].to_numpy()[0])
    X = []
    y = []

    for i, pdb in enumerate(list(PDBs.keys())):
        X.append(featurize(PDBs[pdb], info[i], excelude))
        y.append(df[df['Id'] == pdb]['exp'].to_numpy()[0])

    return train(X, y, excelude)

excelude_list = [0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 18, 19]
dic = {}
for ex in excelude_list:
    hist, res = experiment([ex])
    dic[ex] = [hist, res]

with open('./diff_features.pkl', 'wb') as file:
    # Serialize and write the variable to the file
    pickle.dump(dic, file)

