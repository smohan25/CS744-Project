#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.distributed as dist

import argparse

parser = argparse.ArgumentParser(description="Basic Send and Recv")
parser.add_argument("--rank", type=int, default=0, help="Rank of the process")

args = parser.parse_args()

dist.init_process_group("gloo", init_method='tcp://10.10.1.1:2345',
                        rank=args.rank, world_size=2)

if args.rank == 0:
  t = torch.eye(3, dtype=torch.float)
  t_sp = t.to_sparse()
  dist.send(t_sp, 1)
  print("Tensor sent from 0 to 1")
else:
  t = torch.zeros(3, 3, dtype=torch.float)
  print("Old tensor value", t_sp)
  dist.recv(t_sp, 0)
  print("New tensor value", t_sp)
  print("Tensor recieved at 1 from 0")
