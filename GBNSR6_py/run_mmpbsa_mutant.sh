#!/bin/bash

source /home/ali/Amber/amber22/amber.sh
#-66
$AMBERHOME/bin/MMPBSA.py -O -i ./mutants/mmpbsa.in -o FINAL_RESULTS_MMPBSA.dat \
 -sp mutants/0.15_80_10_pH7.5_6m0j_octbox.BOX.top -cp mutants/6m0j.prmtop -rp mutants/ACE2.prmtop \
 -lp mutants/SARS_CoV_2.prmtop -y mutants/*.crd -mc mutants/6m0j-606.prmtop -ml mutants/SARS-606.prmtop