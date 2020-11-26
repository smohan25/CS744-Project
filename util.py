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

def treeAllReduce(model, rank, size):
  """  Helper function to call tree reduce for the model """
  for param in model.parameters():
    tree_all_reduce(rank, param.data, size)

######################################################
##### END OF TREE ALL REDUCE #########################

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

def butterflyAllReduce(model, rank, size):
  """ Helper function to call butterfly reduce for the model """
  for param in model.parameters():
    butterfly_all_reduce(rank, param.data, size)

######################################################
##### END OF BUTTERFLY ALL REDUCE ####################
    
def topHalf(tensor):
    # In dimension 0
    size = tensor.size(0)
    if size % 2 == 0:
        return tensor.narrow(0, 0, size/2)
    else:
        return tensor.narrow(0, 0, size/2 + 1)
    
def botHalf(tensor):
    size = tensor.size(0)
    if size % 2 == 0:
        return tensor.narrow(0, size/2, size/2)
    else:
        return tensor.narrow(0, size/2+1, size/2)

def recursive_halving_doubling(rank, tensor, size):
  resVec = tensor
  steps = math.log(size, 2)
  d = 2
  for i in range(int(steps)):
    if (rank % d) < d/2:
      dest = rank + d/2
      sendVector = topHalf(resVec)
      recvVector = torch.zeros_like(botHalf(resVec))
      concatVector = torch.zeros_like(sendVector)
      dist.send(sendVector, dest)

      dist.recv(recvVector, dest)

      res = torch.cat((concatVector, recvVector))
      resVec = resVec.add(res)
    else:
      dest = rank - d/2
      sendVector = botHalf(resVec)
      recvVector = torch.zeros_like(topHalf(resVec))
      concatVector = torch.zeros_like(sendVector)
      dist.recv(recvVector, src=dest)

      dist.send(sendVector, dest)

      res = torch.cat((recvVector, concatVector))
      resVec = resVec.add(res)

      
    if (rank % d) < d/2:
      resVec = botHalf(resVec)
    else:
      resVec = topHalf(resVec)
    d *= 2

    # Each node now has 1/size of the total reduce. Perform all-gather...
    d = size
    for i in range(int(steps)):
      recvVector = torch.zeros_like(resVec)
      if (rank % d) < d/2:
        dest = rank + d/2
        dist.send(resVec, dest)
        dist.recv(recvVector, src=dest)
        resVec = torch.cat((recvVector, resVec))
      else:
        dest = rank - d/2
        dist.recv(recvVector, src=dest)
        dist.send(resVec, dest)
        resVec = torch.cat((resVec, recvVector))
      d /= 2

    tensor = resVec/size
        

######################################################
##### END OF REC DOUBLING AND HALVING ################

def getTensorChunk(tensor, i, totalChunks, chunkSize):
    """ Gets the ith chunk of the tensor """
    size = tensor.size(0)
    if i < 0:
        i = totalChunks + i
    # If it's the last chunk, return till end of vector
    if i == totalChunks - 1:
        newSize = size - i*chunkSize
        return tensor.narrow(0, i*chunkSize, newSize)
    return tensor.narrow(0, i*chunkSize, chunkSize)

def getChunkPos(i, totalChunks):
    """ When the chunk pos is -ve, return a +ve val """
    if i < 0:

      return totalChunks + i
    else:
        return i

def ring_all_reduce(rank, tensor, size):
  resVec = tensor
  tensorSize = tensor.size(0)
  totalChunks = size
  chunkSize = tensorSize/totalChunks
  if tensorSize  % size != 0:
    last = tensorSize % size
    
  for i in range(size-1):
    dst = (rank + 1) % size
    sendVector = getTensorChunk(resVec, rank - i, totalChunks, chunkSize)
    send_sig = dist.isend(sendVector, dst=dst)
            
    # Handle the case where the last chunk's size is > the other chunk sizes. eg vector of size 9 and 4 nodes.
    recvNode = rank - 1 if rank - 1 >= 0 else size - 1
    recvChunk = getChunkPos(recvNode - i, totalChunks)
    sizeList = list(resVec.size())
    if recvChunk == totalChunks - 1:
      sizeList[0] = tensorSize - recvChunk*chunkSize
    else:
      sizeList[0] = chunkSize
    recvVec = torch.zeros(sizeList, dtype=sendVector.dtype)

    recv_sig = dist.irecv(recvVec, src=recvNode)
    send_sig.wait()
    recv_sig.wait()
            
    sizeList = list(resVec.size())
    if recvChunk == 0:
      sizeList[0] = tensorSize - chunkSize
      concatVec = torch.zeros(sizeList, dtype=sendVector.dtype)
      recvVec = torch.cat((recvVec, concatVec))
    elif recvChunk == totalChunks - 1:
      sizeList[0] = recvChunk*chunkSize
      concatVec = torch.zeros(sizeList, dtype=sendVector.dtype)
      recvVec = torch.cat((concatVec, recvVec))
    else:
      sizeList[0] = recvChunk*chunkSize
      concatVec1 = torch.zeros(sizeList, dtype=sendVector.dtype)
      sizeList[0] = tensorSize - (recvChunk+1)*chunkSize
      concatVec2 = torch.zeros(sizeList, dtype=sendVector.dtype)
      recvVec = torch.cat((concatVec1, recvVec, concatVec2))

    resVec = resVec.add(recvVec)
             
    # Perform all-gather now....                                                                                                                                                                        
    for i in range(size-1):
      dst = (rank + 1) % size
      sendVector = getTensorChunk(resVec, (rank - i + 1) % size, totalChunks, chunkSize)
      send_sig = dist.isend(sendVector, dst=dst)
      recvNode = rank - 1 if rank - 1 >= 0 else size - 1
      recvChunk = (recvNode - i + 1) %  size
      sizeList = list(resVec.size())
      if recvChunk == totalChunks - 1:
        sizeList[0] = tensorSize - recvChunk*chunkSize
      else:
        sizeList[0] = chunkSize
      recvVec = torch.zeros(sizeList, dtype=sendVector.dtype)

      recv_sig = dist.irecv(recvVec, src=recvNode)
      send_sig.wait()
      recv_sig.wait()

      # Replace received chunk in resVec                                                                                                                                                             
      for i in range(0, recvVec.size(0)):
        resVec[recvChunk*chunkSize + i] = recvVec[i]

      tensor = resVec/size

######################################################
##### END OF RING ALL REDUCE #########################    


def performAllReduce(model, rank, size, topology):
  for param in model.parameters():
    if topology == "rec-double-half":
      recursive_halving_doubling(rank, param.data, size)
    elif topology == "ring":
      ring_all_reduce(rank, param.data, size)
    elif topology == "tree":
      tree_all_reduce(rank, param.data, size)
    elif topology == "butterfly":
      butterfly_all_reduce(rank, param.data, size)

  
