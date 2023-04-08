import pandas as pd
import numpy as np
import pickle
from models.PGCN_model_entropy import get_trained_model, test_model

PDBs = pickle.load(open('../Notebooks/Entropy/PDBs_RDKit.pkl', 'rb'))
df = pd.read_csv('../Notebooks/Entropy/T_data.csv')

K = 4
X_folds = pickle.load(open('X_folds_4.pkl', 'rb'))
y_folds = pickle.load(open('y_folds_4.pkl', 'rb'))

print('Data loaded')

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
    hist, m = get_trained_model(X_train, y_train, epochs=35)
    hists.append(hist)
    test_loss.append(test_model(X_test, y_test, m))

with open('PGCN_K_fold_hists.pkl', 'wb') as file:
    pickle.dump(hists, file)
with open('PGCN_K_fold_test.pkl', 'wb') as file:
    pickle.dump(test_loss, file)
print('finished')