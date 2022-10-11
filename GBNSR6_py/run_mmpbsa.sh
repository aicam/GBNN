#!/bin/bash

source /home/ali/Amber/amber22/amber.sh

$AMBERHOME/bin/MMPBSA.py -O -i mmpbsa.in -o FINAL_RESULTS_MMPBSA.dat \
  -sp ras-raf_solvated.prmtop -cp com.prmtop -rp ras.prmtop -lp raf.prmtop -y single.mdcrd > /dev/null