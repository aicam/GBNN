import numpy as np
import pickle
from models.PGCN_model_entropy import get_trained_model, test_model, data_generator

K = 4
PDBs_folds = pickle.load(open('PDBs_folds_4.pkl', 'rb'))
y_folds = pickle.load(open('y_folds_4.pkl', 'rb'))
x_add_folds = pickle.load(open('x_add_folds_4.pkl', 'rb'))

print('Data loaded')

hists = []
test_loss = []
predicts_train = []
predicts_test = []
for k in range(K):
    X_train = []
    x_add_train = []
    for i in range(K):
        if i != k:
            X_train += PDBs_folds[i]
            x_add_train += x_add_folds[i]
    X_test = PDBs_folds[k]
    x_add_test = x_add_folds[i]
    y_train = np.concatenate(np.array([y_folds[i] for i in range(K) if i != k]), 0)
    y_test = np.array(y_folds[k])
    x_train_parsed = data_generator(X_train, x_add_train)
    x_test_parsed = data_generator(X_test, x_add_test)
    hs, m, x_converted = get_trained_model(x_train_parsed, np.array(y_train), epochs=40)
    predicts_train.append(m.predict(x_converted, batch_size=len(X_train)))
    hists.append(hs)
    k_loss, x_converted = test_model(x_test_parsed, y_test, m)
    test_loss.append(k_loss)
    predicts_test.append(m.predict(x_converted, batch_size=len(X_test)))

with open('PGCN_K_fold_hists.pkl', 'wb') as file:
    pickle.dump(hists, file)
with open('PGCN_K_fold_test.pkl', 'wb') as file:
    pickle.dump(test_loss, file)
with open('PGCN_K_fold_train_pred.pkl', 'wb') as file:
    pickle.dump(predicts_train, file)
with open('PGCN_K_fold_train_pred.pkl', 'wb') as file:
    pickle.dump(predicts_train, file)
print('finished')