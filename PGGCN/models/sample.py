from rdkit import Chem
from PGGCN.models.dcFeaturizer import atom_features as get_atom_features
import numpy as np

def featurize(molecule, info):
    atom_features = []
    for atom in molecule.GetAtoms():
        new_feature = get_atom_features(atom).tolist()
        position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
        new_feature += [atom.GetMass(), atom.GetAtomicNum(), atom.GetFormalCharge()]
        new_feature += [position.x, position.y, position.z]
        for neighbor in atom.GetNeighbors()[:2]:
            neighbor_idx = neighbor.GetIdx()
            new_feature += [neighbor_idx]
        for i in range(2 - len(atom.GetNeighbors())):
            new_feature += [0]
        atom_features.append(np.concatenate([new_feature, info], 0))
    return np.array(atom_features)

sample_guest = Chem.MolFromPDBFile("/home/ali/PycharmProjects/GBNN/Datasets/Mobley/cd-set1/pdb/guest-2.pdb")
sample_host = Chem.MolFromPDBFile("/home/ali/PycharmProjects/GBNN/Datasets/Mobley/cd-set1/pdb/host-acd.pdb")

## this is a sample molecule
sample_complex = Chem.CombineMols(sample_host, sample_guest)
for atom in sample_complex.GetAtoms():
    print(atom.GetSymbol())

## sample info array
info = [1, 2]

## Then it is featurized
sample_features = featurize(sample_complex, info)
print(sample_features)