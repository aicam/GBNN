#!/bin/bash

srun -c 8 --mem=16GB -J default_srun -N 1 -n 1 -G 1 --pty /bin/bash