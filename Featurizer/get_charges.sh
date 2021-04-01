#!/bin/bash
# $1 pdb file name (ex. "ras.pdb")
$AMBERHOME/bin/tleap -s -f $AMBERHOME/dat/leap/cmd/leaprc.protein.ff14SB;
echo "set default PBRadii mbondi";
echo "quit";

