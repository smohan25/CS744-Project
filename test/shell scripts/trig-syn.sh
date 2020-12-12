#!/bin/bash

echo topology $1
echo size $2
echo world size $3
echo density $4
c=`expr $3 - 1`
for i in `seq 1 $c`
do
	echo rank $i
	ssh smohan25@node${i} ./syn-work.sh $i $1 $2 $3 $4&
done
