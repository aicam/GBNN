import numpy as np
from biopandas.pdb import PandasPdb
from deepchem.feat import CircularFingerprint

smile = ''.join(PandasPdb().read_pdb('PDBs/ras-raf.pdb').df['ATOM']['atom_name'].to_numpy())
print(CircularFingerprint().featurize(smile))