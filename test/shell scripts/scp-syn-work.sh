#!/bin/bash
host=$(whoami)
echo "${host}"
for i in `seq 1 $1`
do
	scp ./syn-work.sh ${host}@node${i}:~/syn-work.sh
done
