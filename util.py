import torch
import torch.distributed as dist

import math
from typing import List

def tree_all_reduce(rank: int, tensor: torch.Tensor, world_size: int):
  """
  Supports even number of nodes only.
  """
  # determine the number of rounds of gather, followed by scatter
  rounds = int(math.log2(world_size))
  if 2 ** rounds < world_size:
    rounds += 1
  
  # GATHER
  for i in range(rounds):
    # nodes participating in out and in
    ins = list(range(0, world_size, 2**(i+1)))
    outs = list(set(list(range(0, world_size, 2**i))) - set(ins))

    if rank in outs:
      index = outs.index(rank)
      _in = ins[index]
      dist.send(tensor, _in)

    if rank in ins:
      index = ins.index(rank)
      if index < len(outs):
        _out = outs[index]
        t = torch.zeros_like(tensor)
        dist.recv(t, _out)
        tensor += t

  # at rank 0, divide by world_size to get the average
  if rank == 0:
    tensor /= world_size

  # SCATTER
  for i in range(rounds):
    # nodes participating in out and in
    outs = list(range(0, world_size, 2**(rounds-i)))
    ins = list(set(list(range(0, world_size, 2**(rounds-i-1)))) - set(outs))

    if rank in outs:
      index = outs.index(rank)
      if index < len(ins):
        _in = ins[index]
        dist.send(tensor, _in)
    
    if rank in ins:
      index = ins.index(rank)
      _out = outs[index]
      dist.recv(tensor, _out)


def _my_swap(ins: List, start: int, swap_unit: int):
  for i in range(start, start+swap_unit):
    temp = ins[i]
    ins[i] = ins[i+swap_unit]
    ins[i+swap_unit] = temp


def butterfly_all_reduce(rank: int, tensor: torch.Tensor, world_size: int):
  """
  Supports homogeneous Butterfly networks. Homogeneous networks have 2^n nodes.
  """
  # determine the number of layers
  layers = int(math.log2(world_size))

  # in and out are the same in this algorithm
  for i in range(layers):
    # generate the ins list
    # how many nodes are grouped
    swap_unit = 2 ** i
    # skip
    skip = 2 * swap_unit

    ins = list(range(0, world_size))
    for j in range(0, world_size, skip):
      _my_swap(ins, j, swap_unit)

    # send asynchronously
    send_req = dist.isend(tensor, ins[rank])

    # recv asynchronously
    t = torch.zeros_like(tensor)
    recv_req = dist.irecv(t, ins[rank])

    send_req.wait()
    recv_req.wait()

    # merge
    tensor += t
  
  # divide by world_size to get average
  tensor /= world_size
