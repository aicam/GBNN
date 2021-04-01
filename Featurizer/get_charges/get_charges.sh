#!/bin/bash
# $1 pdb file name (ex. "ras.pdb")

# copy pdb file to the folder
cp $1 ./fake.pdb

# get new pdb with charges
$AMBERHOME/bin/tleap -f tleap_get_charges.in > /dev/null

cat fake_o.pdb

# remove fake pdb files
rm ./fake.pdb
rm ./fake_o.pdb