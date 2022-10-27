# GBNSR6 Python
## Introduction
This library provides functionalities to run GBNSR6 on a trajectory as 
a part of contribution to MMPBSA_py in Amber library. The main functions can be found on Amber
and run them by setting igb=66 in input file (for ex. mmpbsa.in).
<br>
Here we have internal development modules and functions to test the whole system.

## Usage
The basic usage of the library is to run main_test.py on a trajectory
to compare the result of GBNSR6 with MMPBSA (igb=2).
<br>
### Trajectory file
The trajectory file should be in ASCII format. If you have
run the simulation and generated the binary files, just 
convert them to .crd trajectory files.
### MMPBSA input file
The default mmpbsa input file is stored at mmpbsa.in, please
do not remove startframe and endframe arguments as the
program need them to modify and replace.
### GBNSR6 input file
You can modify GBNSR6 input file in gbnsr6.in. No additional
arguments are required to run the program.
### Help
```shell
usage: main_test.py [-h] [-sp <Topology File>] [-rp <Topology File>] [-lp <Topology File>] [-cp <Topology File>] [-is Bool] [--traj-path Path] [--traj-type file type]

GBNSR6 python version running over trajectory

optional arguments:
  -h, --help            show this help message and exit

Miscellaneous Options:
  -sp <Topology File>   Topology file of a fully solvated system. If provided, the atoms specified by <strip_mask> will be stripped from the trajectory file. The complex
                        topology file (-cp) must be consistent with this stripped trajectory
  -rp <Topology File>   Topology file of the unbound receptor. If omitted (and -lp is omitted, too), a stability calculation with just the complex will be performed.
  -lp <Topology File>   Topology file of the unbound ligand. If omitted (and -rp is omitted, too), a stability calculation with just the complex will be performed.
  -cp <Topology File>   Topology file of the bound complex (or the single system for 'stability' calculations)
  -is Bool              Is the simulation run over a solvated structure?
  --traj-path Path      Trajectory files path
  --traj-type file type
                        Trajectory files type

```
