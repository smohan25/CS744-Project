import torch
import torch.distributed as dist

import math

def treeAllReduce(rank: int, tensor: torch.Tensor, world_size: int):
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
