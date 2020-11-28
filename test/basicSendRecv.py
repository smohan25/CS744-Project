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
  t = torch.tensor([[1,0], [0,1]])
  dist.send(t, 1)
  print("Tensor sent from 0 to 1")
else:
  t = torch.tensor([[0,0], [0,0]])
  print("Old tensor value", t)
  dist.recv(t, 0)
  print("New tensor value", t)
  print("Tensor recieved at 1 from 0")
