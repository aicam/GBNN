import tensorflow as tf
import numpy as np
import pandas as pd
# !curl -Lo conda_installer.py https://raw.githubusercontent.com/deepchem/deepchem/master/scripts/colab_install.py
import conda_installer

# conda_installer.install()
conda_installer.install()
# !/root/miniconda/bin/conda info -e
import rdkit

from deepchem.feat.graph_features import atom_features as get_atom_features

df = pd.read_csv('Datasets/Mobley/info.csv')
df = df.dropna()
df = df[df['Dataset Name'] == 'cd-set1']

training_cols = [col for col in df.columns if (col[:3] == 'gb_' and not col.__contains__('Etot') and not col.__contains__('Ex_') and not col.__contains__('delta')) or (col.__contains__('VDWAALS'))]
# training_cols = ['gb_Ex_difference']

PDBs = {}
from os import listdir
from os.path import isfile, join
mypath = 'Datasets/Mobley/cd-set1/pdb/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
host_name = [x for x in onlyfiles if x.__contains__('host')][0]
host_mol = rdkit.Chem.rdmolfiles.MolFromPDBFile(mypath + '/' + host_name)
onlyfiles.remove(host_name)
for f in onlyfiles:
    guest_mol = rdkit.Chem.rdmolfiles.MolFromPDBFile(mypath + '/' + f)
    PDBs.update({f.split('.')[0] : rdkit.Chem.CombineMols(host_mol,guest_mol)})


def featurize(molecule):
    def convertBONDtoOneHotVector(bond_type):
        BOND_DICT = {dir(rdkit.Chem.rdchem.BondType)[i]: i for i in range(22)}
        one_hot = np.zeros(22)
        one_hot[BOND_DICT[str(bond_type)]] = 1
        return one_hot

    atom_features = []
    adjacency_list = []
    for atom in molecule.GetAtoms():
        new_feature = get_atom_features(atom).tolist()
        position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
        new_feature += [atom.GetMass(), atom.GetAtomicNum()]
        new_feature += [position.x, position.y, position.z]
        atom_features.append(new_feature)
    atom_features = np.array(atom_features)
    for atom in molecule.GetAtoms():
        atom_bonds = []
        for neighbor in atom.GetNeighbors():
            bond_type = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType()
            neighbor_idx = neighbor.GetIdx()
            atom_bonds.append([neighbor_idx, convertBONDtoOneHotVector(bond_type)])
        adjacency_list.append(atom_bonds)
    return [atom_features, adjacency_list]

X = []
y = []
info = []
for pdb in list(PDBs.keys()):
    X.append(featurize(PDBs[pdb]))
    info.append(df[df['Guest'] == pdb.replace('-s', '-')][training_cols].to_numpy()[0])
    y.append(df[df['Guest'] == pdb.replace('-s', '-')]['EX _H_(kcal/mol)'].abs().to_numpy()[0])
info = np.array(info)

import layers


class PGGCNModel(tf.keras.Model):
    def __init__(self, num_atom_features=80, r_out_channel=40, c_out_channel=1024):
        super().__init__()
        self.ruleGraphConvLayer = layers.RuleGraphConvLayer(r_out_channel, num_atom_features)
        self.ruleGraphConvLayer.combination_rules = []
        self.conv = layers.ConvLayer(c_out_channel)
        self.dense1 = tf.keras.layers.Dense(300, activation='sigmoid')
        self.dense2 = tf.keras.layers.Dense(100, activation='sigmoid')
        self.dense3 = tf.keras.layers.Dense(20, activation='softmax')
        # self.dense4 = tf.keras.layers.Dense(150, activation='relu')
        # self.dense5 = tf.keras.layers.Dense(50, activation='relu')
        self.dense6 = tf.keras.layers.Dense(1, activation='relu')
        self.dense7 = tf.keras.layers.Dense(1,
                                            kernel_initializer=tf.keras.initializers.Constant(
                                                [.5, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1]),
                                            bias_initializer=tf.keras.initializers.Zeros(),
                                            activation=tf.keras.activations.relu)

    def set_physics_info(self, info):
        self.physics_info = info.reshape([-1, 15])

    def addRule(self, rule, start_index, end_index=None):
        self.ruleGraphConvLayer.addRule(rule, start_index, end_index)

    def set_input(self, all_inputs):
        self.all_inputs = all_inputs

    def set_adjacency_list(self, a_l):
        self.a_l = a_l

    def set_input_shapes(self, i_s):
        self.i_s = i_s

    def call(self, inputs):
        x_a = []
        for i in range(len(self.i_s)):
            x_a.append(inputs[i][:self.i_s[i]])
        #         agg = [[x_a, self.a_l]]
        agg = []
        for i in range(len(x_a)):
            agg.append([x_a[i], self.a_l[i]])
        #         agg = self.all_inputs
        #         inputs = self.all_inputs
        x = self.ruleGraphConvLayer(agg)
        x = self.conv(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        # x = self.dense4(x)
        # x = self.dense5(x)
        model_var = self.dense6(x)
        merged = tf.concat([model_var, self.physics_info], axis=1)
        out = self.dense7(merged)
        return out


m = PGGCNModel()
m.addRule("sum", 0, 75)
m.addRule("multiply", 75, 77)
m.addRule("distance", 77, 80)
m.set_physics_info(info)

import keras.backend as K


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred[0] - y_true))) + (1 / K.mean(.5 + y_pred[1]))


def pure_rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred[0] - y_true)))


m.compile(loss=pure_rmse, optimizer='adam')
X_atoms = []
X_adj = []
for x in X:
    X_atoms.append(x[0])
    X_adj.append(x[1])
input_shapes = []
for i in range(len(X_atoms)):
    input_shapes.append(np.array(X_atoms[i]).shape[0])
m.set_input_shapes(input_shapes)
for i in range(len(X_atoms)):
    if X_atoms[i].shape[0] < 77:
        new_list = X_atoms[i].tolist()
        for j in range(80 - X_atoms[i].shape[0]):
            new_list.append([0.0] * 80)
        #         new_list.append(np.concatenate((info[i], [0]*65)))
        X_atoms[i] = np.array(new_list)
m.set_adjacency_list(X_adj)
m.set_physics_info(info)
X_atoms = np.array(X_atoms)
y = np.array(y)
hist = m.fit(X_atoms, y, epochs=20)
import pickle
with open('res.pkl', 'wb') as file:
    pickle.dump(hist.history, file)