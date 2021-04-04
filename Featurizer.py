import argparse
import os
from os.path import isfile, join
from Featurizer.main import get_pdb_dataframe


try:
    amberhome = os.environ.get('AMBERHOME')
except Exception:
    exit("AMBERHOME environment variable not set")

parser = argparse.ArgumentParser(description="Extract effective born radii & charges and distances from PDB files for GBNN model")
parser.add_argument('-i', help='Input directory cotains PDB files')
parser.add_argument('-o', help='Output directory which HDF5 files will be added to')

args = parser.parse_args()
PDB_directory = args.i
H5_directory = args.o

PDBs = [f for f in os.listdir(PDB_directory) if isfile(join(PDB_directory, f))]


for pdb in PDBs:
    os.chdir("Featurizer")
    df = get_pdb_dataframe('../../' + PDB_directory + '/' + pdb)
    os.chdir('..')
    df.to_hdf(H5_directory + '/' + pdb.split('.')[0] + ".h5", pdb.split('.')[0])
