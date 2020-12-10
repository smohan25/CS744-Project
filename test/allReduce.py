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
parser.add_argument("--density", help="Density of tensor", default=0.001)
parser.add_argument("--size", help="Size of tensor", default=1000)

args = parser.parse_args()
world_size = args.world_size

# Modify IP accordingly.
dist.init_process_group("gloo", init_method='tcp://10.128.0.2:2345',
                        rank=args.rank, world_size=world_size)

size = int(args.size)
t = torch.tensor(sparse.random(size, size, density=args.density).A, dtype=torch.float) \
    * np.random.randint(10, 100)

t_ns = t.detach().clone()
t_ns1 = t.detach().clone()

# AllReduce
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


# AllGather sparse
t2 = Timer(2)
t2.start()
t_ns1 = t_ns1.to_sparse()#.coalesce()
nnz = t_ns1._nnz()
indices = t_ns1.indices()
values = t_ns1.values()

nnz_tensor = torch.tensor(nnz)
dist.all_reduce(nnz_tensor, op=dist.ReduceOp.SUM)
gatherListIndices = [torch.zeros([2, nnz], dtype=torch.long) for _ in range(world_size)]
dist.all_gather(gatherListIndices, indices)
gatherListValues = [torch.zeros([nnz], dtype=torch.float) for _ in range(world_size)]
dist.all_gather(gatherListValues, values)

dic = {}
for i, v in zip(gatherListIndices, gatherListValues):
    for x, y, z in zip(i[0], i[1], v):
        x, y, z = x.item(), y.item(), z.item()
        if (x, y) in dic:
            dic[(x, y)] += z
        else:
            dic[(x, y)] = z
new_indices = [[], []]
new_values = []
for k, v in dic.items():
    new_indices[0].append(k[0])
    new_indices[1].append(k[1])
    new_values.append(v)
#print(new_indices)
sparseTensor = torch.sparse.LongTensor(torch.LongTensor(new_indices), torch.FloatTensor(new_values), t.size())
e2 = t2.stop()

print("Allreduce, Allgather, Allgather-sparse")
print(f"{e0:0.4f}, {e1:0.4f}, {e2:0.4f}")
