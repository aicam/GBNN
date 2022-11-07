#!/bin/bash

source /home/ali/Amber/amber22/amber.sh
#-66
$AMBERHOME/bin/MMPBSA.py -O -i mmpbsa.in -o FINAL_RESULTS_MMPBSA.dat -sp prods/complex_solvated.prmtop \
  -cp tmpdata/6m0j.prmtop -rp tmpdata/ACE2.prmtop -lp tmpdata/SARS_CoV_2.prmtop -y prods/*.crd