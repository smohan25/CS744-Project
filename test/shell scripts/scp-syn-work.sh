#!/bin/bash
for i in `seq 1 $1`
do
	scp ./syn-work.sh agholba@node${i}:~/syn-work.sh
done
