from Featurizer.main import get_pdb_dataframe
import os
print(os.path.dirname(os.path.abspath(__file__)))
a = get_pdb_dataframe(os.path.dirname(os.path.abspath(__file__)) + '/Datasets/pdbbind_complex/1a1e.pdb')
# print(a)
