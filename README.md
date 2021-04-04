# About project
This project has two parts: first is the featurizer which extract "partial charges", "effective Born radii" and "Distances" between atoms from PDB files.
This featurizers is being used to generate pandas dataframes with hdf5 format instead of PDB raw data. The second part is a deep learning model based on 
Grid-Based Surface Generalized Born Model for calculating free binding energy between protein and ligand. <br><br>
This library uses Amber as the application to featurize pdb files, in this regard, $AMBERHOME environment variable should be set. Furthermore, because Amber does not 
provide API from python, you can find bash scripts under Featurizer folder which simulate user in Amber to extract information.

# Getting started
At the beginning we need to generate our dataset by featurizer and next we can train our model and test.
First, clone the library:
```SH
git clone https://github.com/aicam/GBNN
```
You need to give executable permission to bash script files:
```SH
cd Featurizer
sudo chmod +x get_born/get_born.sh
sudo chmod +x get_charges/get_charges.sh
```
Next, you should place your PDB files under root directory for example PDBs and create a directory for output dataframes to be stored at.
## Featurizer
Featurizer script will crawl over all pdb files placed on the pdb directory (-i option) and create a .h5 dataframe in the output directory (-o option) for each one.
```SH
python3 Featurizer.py -i <PDB files directory name under the root> -o <Output directory name under the root>
```
### Atoms partial charges
Partial charges are being calculated by Amber and PDB directly. Script will run tleap from Amber and call <b>set default PdbWriteCharges on</b> and <b>savePdb</b> 
command to store the charges.

### Atoms effective Born radii
Effective born radii has the same procedure as atoms partial charges with tleap and <b>solvatebox TIP3PBOX 12.0</b> with the following properties in Amber:
```
compute
&cntrl
inp=1
/
&gb
epsin=1.0, epsout=78.5, istrng=0, dprob=1.4, space=0.5,
arcres=0.2, B=0.028, alpb=1, rbornstat=1, cavity_surften=0.005
/

```
### Atom distances (R matrix)
This function is being performed by Tensorflow (required library) and parallel loop to find M nearest atoms for each atom in PDB file. The distance has been
defined in the <b>AtomDistance(x, y)</b> which is based on Euclidean distance and you can replace it with your atom distance function.

Note: featurizer.py needs to move over directories (os.chdir), therefore don't change the files order
