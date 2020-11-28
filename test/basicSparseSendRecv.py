#!/usr/bin/env python3
# coding: utf-8

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist

from util import send_sparse, recv_sparse

import argparse

parser = argparse.ArgumentParser(description="Basic Sparse Send and Recv")
parser.add_argument("--rank", type=int, default=0, help="Rank of the process")
parser.add_argument("--master", type=str, default='10.10.1.1:2345',
                    help="Master IP")

args = parser.parse_args()

dist.init_process_group("gloo", init_method=f'tcp://{args.master}',
                        rank=args.rank, world_size=2)

if args.rank == 0:
  t = torch.eye(3, dtype=torch.float)
  s1, s2, s3 = send_sparse(t, 1)
  s1.wait()
  s2.wait()
  s3.wait()
  print("Tensor sent from 0 to 1")
else:
  t = torch.zeros(3, 3, dtype=torch.float)
  print("Old tensor value", t)
  t = recv_sparse(0, 2, [3, 3], dtype=torch.float)
  print("New tensor value", t)
  print("Tensor recieved at 1 from 0")
