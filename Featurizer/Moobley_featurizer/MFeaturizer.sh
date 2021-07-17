#!/bin/bash
cp $1 ./fake_guest.mol2
$AMBERHOME/bin/antechamber -i ./fake_guest.mol2 -fi mol2 -dr no -o guest-fake.prep -fo prepi -c bcc
$AMBERHOME/bin/parmchk2 -i guest-fake.prep -f prepi -o guest-fake.frcmod

export host_name
guest_name=guest-fake
export guest_name
complex=$host_name-guest-fake
#update guest and host name in tleap input file and save it as tleap-new.in
cd ../../..
sed -e "s/host_name/$host_name/g;s/guest/$guest_name/g;s/complex_name/$complex/g" tleap.in > tleap-new.in
cdf
cd $host_name-guest-$i
#sed 's/guest_name/$guest-$i/g' ../../../tleap.in
$AMBERHOME/bin/tleap -f ../../../tleap-new.in
mkdir gbnsr6-$host_name-guest-$i
cd gbnsr6-$host_name-guest-$i
$AMBERHOME/bin/gbnsr6 -O -i ../../../../gbnsr6.in -p ../$host_name.prmtop -c ../$host_name.inpcrd -o $host_name.out
tail -n 15 $host_name.out | grep -E -o 'EGB.{0,21}|ESURF.{0,21}|EELEC.{0,20}|Etot.{0,20}|1-4 EEL.{2,18}' >> $host_name-gb.txt
$AMBERHOME/bin/gbnsr6 -O -i ../../../../gbnsr6.in -p ../$complex.prmtop -c ../$complex.inpcrd -o $complex.out
tail -n 15 $complex.out | grep -E -o 'EGB.{0,21}|ESURF.{0,21}|EELEC.{0,20}|Etot.{0,20}|1-4 EEL.{2,18}' >> $complex-gb.txt
$AMBERHOME/bin/gbnsr6 -O -i ../../../../gbnsr6.in -p ../guest-$i.prmtop -c ../guest-$i.inpcrd -o guest-$i.out
tail -n 15 guest-$i.out | grep -E -o 'EGB.{0,21}|ESURF.{0,21}|EELEC.{0,20}|Etot.{0,20}|1-4 EEL.{2,18}' >> guest-$i-gb.txt