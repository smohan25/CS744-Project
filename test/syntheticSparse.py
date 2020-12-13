#!/usr/bin/env python
# coding: utf-8

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist

import argparse

from util import _performAllReduce
from timer import Timer

import scipy.sparse as sparse
import numpy as np

parser = argparse.ArgumentParser(description="Basic Tree AllReduce")
parser.add_argument("--rank", type=int, default=0, help="Rank of the process")
parser.add_argument("--world_size", type=int,default=2, help="No. of processes")
parser.add_argument("--topology", type=str, choices=["tree", "butterfly", 
                    "ring", "rec-double-half"], help="Choose topology to test")
parser.add_argument("--size", type=str, default="1000,1000", 
                    help="The size of random tensor")
parser.add_argument("--density", type=float, default=0.01, 
                    help="The density of sparse tensor")

args = parser.parse_args()

np.random.seed(args.rank)

dist.init_process_group("gloo", init_method='tcp://10.10.1.1:2345',
                        rank=args.rank, world_size=args.world_size)

size = [int(x) for x in args.size.split(',')]

# generate a random sparse tensor of size 'size'
# t = torch.tensor(sparse.random(size[0], size[1], density=args.density).A, dtype=torch.float) \
#     * np.random.randint(10, 100)
t = torch.tensor(sparse.random(size[0], size[1]).A, dtype=torch.float) \
    * np.random.randint(10, 100)

t_1 = torch.tensor(sparse.random(size[0], size[1], density=0.001).A, dtype=torch.float) \
    * np.random.randint(10, 100)

# clone for non-sparse
t_ns = t.detach().clone()
# print("before t_ns", t_ns)
# perform non-sparsified allReduce
t0 = Timer(0)
t0.start()
_performAllReduce(t_ns, args.rank, args.world_size, args.topology, False)
e0 = t0.stop()
# print("after t_ns", t_ns)
# perform sparsified allReduce
# print("before t", t)
t1 = Timer(1)
t1.start()
e2 = _performAllReduce(t, args.rank, args.world_size, args.topology, True)
e1 = t1.stop()

t3 = Timer(3)
t3.start()
e4 = _performAllReduce(t_1, args.rank, args.world_size, args.topology, True)
e3 = t3.stop()
print(f"{e0:0.4f}, {e1:0.4f}, {e2:0.4f}, {e3:0.4f}, {e4:0.4f}")
# print("after t", t)
