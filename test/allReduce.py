#!/usr/bin/env python3                                                                               # coding: utf-8

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist

import argparse
import scipy.sparse as sparse
import numpy as np
from timer import Timer

parser = argparse.ArgumentParser(description="Basic Tree AllReduce")
parser.add_argument("--rank", type=int, default=0, help="Rank of the process")
parser.add_argument("--world_size", type=int,default=2, help="No. of processes")
parser.add_argument("--sparse", help="Test sparse", default=False, action="store_true")

args = parser.parse_args()
world_size = args.world_size

dist.init_process_group("gloo", init_method='tcp://10.10.1.1:2345',
                        rank=args.rank, world_size=world_size)

t = torch.tensor(sparse.random(1000, 1000).A, dtype=torch.float) \
    * np.random.randint(10, 100)

t_ns = t.detach().clone()

#AllReduce
t0 = Timer(0)
t0.start()
dist.all_reduce(t, op=dist.ReduceOp.SUM)
t /= world_size
e0 = t0.stop()

# AllGather
gatherList = [torch.zeros_like(t) for _ in range(args.world_size)]
t1 = Timer(1)
t1.start()
dist.all_gather(gatherList, t_ns)
res = gatherList[0]
for i in range(1, len(gatherList)):
    res = res.add(gatherList[i])
res /= world_size
e1 = t1.stop()

print(f"{e0:0.4f}, {e1:0.4f}")
