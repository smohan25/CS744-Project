#!/bin/bash
cd CS744-Project
source ./venv/bin/activate
cd test
#echo rank $2
#echo topology $2
#echo size $3
#echo world size $4
./syntheticSparse.py --rank $1 --topology $2 --size $3 --world_size $4
