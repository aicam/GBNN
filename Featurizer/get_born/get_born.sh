#!/bin/bash
# $1 pdb file name (ex. "ras.pdb")

# copy pdb file to the folder
cp $1 ./fake.pdb

# get new pdb with charges
$AMBERHOME/bin/tleap -f tleap_get_files.in

# generate gbnsr6 born radii
$AMBERHOME/bin/gbnsr6 -i gbnsr6.in -o fake_o -p fake.prmtop -c fake.inpcrd

cat fake_o
# remove fake pdb files
rm ./fake.pdb
rm ./fake_o
rm ./fake.prmtop
rm ./fake.inpcrd