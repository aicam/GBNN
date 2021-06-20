#!/bin/bash
cp $1 ./guest-1.mol2
cp $2 ./host-cb7.mol2



$AMBERHOME/bin/antechamber -i guest-1.mol2 -fi mol2 -o g1.prep -fo prepi -c bcc
$AMBERHOME/bin/parmchk2 -i g1.prep -f prepi -o g1.frcmod

$AMBERHOME/bin/tleap -f tleap_gbnsr6.in
$AMBERHOME/bin/gbnsr6 -O -i gbnsr6.in -p cb7-solvated.prmtop -c cb7-solvated.inpcrd -o cb7-gbout
$AMBERHOME/bin/gbnsr6 -O -i gbnsr6.in -p g1-solvated.prmtop -c g1-solvated.inpcrd -o g1-gbout
$AMBERHOME/bin/gbnsr6 -O -i gbnsr6.in -p cb7-g1-solvated.prmtop -c cb7-g1-solvated.inpcrd -o cb7-g1-gbout