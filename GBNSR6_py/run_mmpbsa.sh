#!/bin/bash

source /home/ali/Amber/amber22/amber.sh

$AMBERHOME/bin/MMPBSA.py -O -i mmpbsa.in -o FINAL_RESULTS_MMPBSA.dat \
  -cp prods/complex.prmtop -rp prods/receptor.prmtop -lp prods/ligand.prmtop -y prods/*.crd