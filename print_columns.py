#!/usr/bin/env python

import numpy as np

f = open("out", 'r')
lines = f.readlines()
f.close()

arr = []
for line in lines:
  _arr = line.split(',')
  for i in range(len(_arr)):
    _arr[i] = _arr[i].strip()
  arr.append(_arr)

# print(arr)

cols = len(arr[0])
arr = np.array(arr)
for col in range(cols):
  _arr = arr[:, col]
  for i in _arr:
    print(i)
  print()
