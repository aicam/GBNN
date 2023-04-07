import pandas as pd
import numpy as np
from deepchem.feat.graph_features import atom_features as get_atom_features
import pickle
from models.PGCN_model_entropy import get_trained_model, test_model
from models.dcFeaturizer import atom_features as get_atom_features

PDBs = pickle.load(open('../Notebooks/Entropy/PDBs_RDKit.pkl', 'rb'))
df = pd.read_csv('../Notebooks/Entropy/T_data.csv')

print('Data collected')

info = []
for pdb in list(PDBs.keys()):
    info.append(df[df['Id'] == pdb][['TS_comp', 'TS_host', 'TS_ligand']].to_numpy()[0])



def featurize(molecule, info):
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
        atom_features.append(np.concatenate([new_feature, info], 0))
    return np.array(atom_features)

X = []
y = []
for i, pdb in enumerate(list(PDBs.keys())):
    X.append(featurize(PDBs[pdb], info[i]))
    y.append(df[df['Id'] == pdb]['exp'].to_numpy()[0])

print('Input data prepared')

K = 4
fold_size = len(X) // K
X_folds = [X[i*fold_size:(i + 1)*fold_size] for i in range(K)]
y_folds = [y[i*fold_size:(i + 1)*fold_size] for i in range(K)]

hists = []
test_loss = []
for k in range(K):
    print('Running fold %d' % k)
    X_train = []
    for i in range(K):
        if i != k:
            X_train += X_folds[i]
    X_test = X_folds[k]
    y_train = np.concatenate(np.array([y_folds[i] for i in range(K) if i != k]), 0)
    y_test = np.array(y_folds[k])
    hist, m = get_trained_model(X_train, y_train, epochs=20)
    hists.append(hist)
    test_loss.append(test_model(X_test, y_test, m))

with open('PGCN_K_fold_hists.pkl', 'wb') as file:
    pickle.dump(hists, file)
with open('PGCN_K_fold_test.pkl', 'wb') as file:
    pickle.dump(test_loss, file)
print('finished')