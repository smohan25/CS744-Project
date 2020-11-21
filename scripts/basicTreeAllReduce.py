#!/usr/bin/env python3
# coding: utf-8

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist

import argparse

from util import tree_all_reduce

parser = argparse.ArgumentParser(description="Basic Tree AllReduce")
parser.add_argument("--rank", type=int, default=0, help="Rank of the process")
parser.add_argument("--world_size", type=int, default=2, help="No. of processes")

args = parser.parse_args()
world_size = args.world_size

dist.init_process_group("gloo", init_method='tcp://10.10.1.1:2345',
                        rank=args.rank, world_size=world_size)

t = torch.eye(2) * (args.rank + 1)

print("Old tensor at", args.rank, "was", t)
  
tree_all_reduce(args.rank, t, world_size)

print("New tensor at", args.rank, "is", t)
