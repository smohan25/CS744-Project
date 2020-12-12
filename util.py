import torch
import torch.distributed as dist

import math
from typing import List


def send_sparse(tensor, dst, sparse=None):
    """ Sends a sparse tensor. To send a sparse tensor we need to communicate 3 things.
    1. nnz - number of non-zero elements.
    2. indices - the position of these non-zero elements.
    3. values - the values of these non-zero elements.

    Params: tensor - tensor to send
    dst - destination node

    Return: 3 signals to wait on. Need to explicitly call wait() on them.
    """

    if not sparse:
        # Convert to a sparse tensor
        tensor = tensor.to_sparse().coalesce()

    send_nnz = dist.isend(torch.tensor(tensor._nnz()), dst)
    send_indices = dist.isend(tensor.indices(), dst)
    send_values = dist.isend(tensor.values(), dst)

    return send_nnz, send_indices, send_values


def recv_sparse(src, tensorDim, tensorSize, dtype, s1=None, s2=None, s3=None, sparse=None):
    """ Receives a sparse tensor asynchronously from a node.
    Pass s1, s2, s3 when 2 nodes are both sending and receiving.
    These correspond to the respective signals from the send_sparse fucntion.

    Params: src - the sender node
    tensorDim - the dimension of the tensor to be received.
    tensorSize - the size of the tensor.
    dtype - type of tensor (eng: int)
    s1, s2, s3 - sent signals of the vector this node sent.
    If not provided, simply receive a tensor.

    Returns: Dense representation of the received vector.
    """

    nnz = torch.zeros([1], dtype=int)
    recv_nnz = dist.irecv(nnz, src)

    if s1 is not None:
        s1.wait()
    recv_nnz.wait()

    indices = torch.zeros([tensorDim, nnz], dtype=int)
    recv_indices = dist.irecv(indices, src)

    values = torch.zeros([nnz], dtype=dtype)
    recv_values = dist.irecv(values, src)

    if s2 is not None:
        s2.wait()
    recv_indices.wait()

    if s3 is not None:
        s3.wait()
    recv_values.wait()

    recvSparseTensor = torch.sparse.LongTensor(
        indices, values, torch.Size(tensorSize))
    if sparse:
        return recvSparseTensor
    else:
        return recvSparseTensor.to_dense()


def tree_all_reduce(rank: int, tensor: torch.Tensor, world_size: int,
                    sparse: bool):
    """
    Supports even number of nodes only.
    """
    # determine the number of rounds of gather, followed by scatter
    rounds = int(math.log2(world_size))
    if 2 ** rounds < world_size:
        rounds += 1

    from timer import Timer
    if sparse:
        tensor = tensor.to_sparse()
        
    # GATHER
    for i in range(rounds):
        # nodes participating in out and in
        ins = list(range(0, world_size, 2**(i+1)))
        outs = list(set(list(range(0, world_size, 2**i))) - set(ins))

        if rank in outs:
            index = outs.index(rank)
            _in = ins[index]
            if sparse:
                s1, s2, s3 = send_sparse(tensor.coalesce(), _in, sparse=True)
                s1.wait()
                s2.wait()
                s3.wait()
            else:
                dist.send(tensor, _in)

        if rank in ins:
            index = ins.index(rank)
            if index < len(outs):
                _out = outs[index]
                if sparse:
                    t = recv_sparse(_out, len(tensor.size()), tensor.size(),
                                    torch.float, s1=None, s2=None, s3=None, sparse=True)
                    tensor = tensor.add(t)
                else:
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
                if sparse:
                    s1, s2, s3 = send_sparse(tensor.coalesce(), _in, sparse=True)
                    s1.wait()
                    s2.wait()
                    s3.wait()
                else:
                    dist.send(tensor, _in)

        if rank in ins:
            index = ins.index(rank)
            _out = outs[index]
            if sparse:
                t = recv_sparse(_out, len(tensor.size()), tensor.size(),
                                    torch.float, s1=None, s2=None, s3=None, sparse=True)
                tensor.copy_(t)
            else:
                dist.recv(tensor, _out)


def treeAllReduce(model, rank, size):
    """  Helper function to call tree reduce for the model """
    for param in model.parameters():
        tree_all_reduce(rank, param.data, size)

######################################################
##### END OF TREE ALL REDUCE #########################


def _swap(ins: List, start: int, swap_unit: int):
    for i in range(start, start+swap_unit):
        temp = ins[i]
        ins[i] = ins[i+swap_unit]
        ins[i+swap_unit] = temp


def butterfly_all_reduce(rank: int, tensor: torch.Tensor, world_size: int,
                        sparse: bool):
    """
    Supports homogeneous Butterfly networks.
    Homogeneous networks have 2^n nodes.
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
            _swap(ins, j, swap_unit)

        if sparse:
            # send asynchronously
            s1, s2, s3 = send_sparse(tensor, ins[rank])

            # recv asynchronously
            t = recv_sparse(ins[rank], len(tensor.size()), tensor.size(),
                            torch.float, s1=s1, s2=s2, s3=s3)
        
        else:
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
        return tensor.narrow(0, 0, int(size/2))
    else:
        return tensor.narrow(0, 0, int(size/2) + 1)


def botHalf(tensor):
    size = tensor.size(0)
    if size % 2 == 0:
        return tensor.narrow(0, int(size/2), int(size/2))
    else:
        return tensor.narrow(0, int(size/2) + 1, int(size/2))


def recursive_halving_doubling(rank, tensor, size, sparse):
    """ Dim 0 of tensor must be divisible by the size (number of nodes in the cluster)
    No support for otherwise """
    
    resVec = tensor
    steps = math.log(size, 2)
    tensorDim = len(tensor.size())
    d = 2
    for i in range(int(steps)):
        start_index = 0
        send_signal, recv_signal = None, None
        if (rank % d) < d/2:
            dest = int(rank + d/2)
            sendVector = topHalf(resVec)
            recvVector = torch.zeros_like(botHalf(resVec))

            if sparse:
                s1, s2, s3 = send_sparse(sendVector, dest)
                recvVector = recv_sparse(
                    dest, tensorDim, list(recvVector.size()), sendVector.dtype, s1, s2, s3)
            else:
                send_signal = dist.isend(sendVector, dest)
                recv_signal = dist.irecv(recvVector, dest)
            start_index = int(size/2)
        else:
            dest = int(rank - d/2)
            sendVector = botHalf(resVec)
            recvVector = torch.zeros_like(topHalf(resVec))
            
            if sparse:
                s1, s2, s3 = send_sparse(sendVector, dest)
                recvVector = recv_sparse(
                    dest, tensorDim, list(recvVector.size()), sendVector.dtype, s1, s2, s3)
            else:
                send_signal = dist.isend(sendVector, dest)
                recv_signal = dist.irecv(recvVector, dest)

        if not sparse:
            send_signal.wait()
            recv_signal.wait()
        index = 0
        for j in range(start_index, recvVector.size(0)):
            recvVector[index] = recvVector[index].add(resVec[j])
            index += 1

        resVec = recvVector
        """if (rank % d) < d/2:
            resVec = botHalf(resVec)
        else:
            resVec = topHalf(resVec)
        """
        d *= 2

    # Each node now has 1/size of the total reduce. Perform all-gather...
    d = size
    for i in range(int(steps)):
        send_signal, recv_signal = None, None
        recvVector = torch.zeros_like(resVec)
        if (rank % d) < d/2:
            dest = int(rank + d/2)
            if sparse:
                s1, s2, s3 = send_sparse(resVec, dest)
                recvVector = recv_sparse(
                    dest, tensorDim, list(recvVector.size()), resVec.dtype, s1, s2, s3)
            else:
                send_signal = dist.isend(resVec, dest)
                recv_signal = dist.irecv(recvVector, src=dest)
                send_signal.wait()
                recv_signal.wait()
            resVec = torch.cat((recvVector, resVec))
        else:
            dest = int(rank - d/2)
            if sparse:
                s1, s2, s3 = send_sparse(resVec, dest)
                recvVector = recv_sparse(
                    dest, tensorDim, list(recvVector.size()), resVec.dtype, s1, s2, s3)
            else:
                send_signal = dist.isend(resVec, dest)
                recv_signal = dist.irecv(recvVector, src=dest)
                send_signal.wait()
                recv_signal.wait()
            resVec = torch.cat((resVec, recvVector))
        d /= 2

    resVec /= size
    tensor.copy_(resVec)


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


def ring_all_reduce(rank, tensor, size, sparse):
    resVec = tensor
    tensorSize = tensor.size(0)
    tensorDim = len(tensor.size())
    totalChunks = size
    chunkSize = int(tensorSize/totalChunks)
    if tensorSize % size != 0:
        last = tensorSize % size

    lastChunk = getTensorChunk(resVec, totalChunks - 1, totalChunks, chunkSize)
    lastChunk_sparse = lastChunk.to_sparse()

    for i in range(size-1):
        dst = (rank + 1) % size
        sendVector = getTensorChunk(resVec, rank - i, totalChunks, chunkSize)
        if sparse:
            s1, s2, s3 = send_sparse(sendVector, dst)
        else:
            send_sig = dist.isend(sendVector, dst=dst)

        # Handle the case where the last chunk's size is > the other chunk sizes. eg vector of size 9 and 4 nodes.
        recvNode = rank - 1 if rank - 1 >= 0 else size - 1
        recvChunk = getChunkPos(recvNode - i, totalChunks)

        sizeList = list(resVec.size())
        if recvChunk == totalChunks - 1:
            sizeList[0] = tensorSize - recvChunk*chunkSize
        else:
            sizeList[0] = chunkSize

        if sparse:
            recvVec = recv_sparse(recvNode, tensorDim,
                                  sizeList, sendVector.dtype, s1, s2, s3)
        else:
            recvVec = torch.zeros(sizeList, dtype=sendVector.dtype)
            recv_sig = dist.irecv(recvVec, src=recvNode)

        if not sparse:
            send_sig.wait()
            recv_sig.wait()

        index = 0
        for j in range(recvChunk*chunkSize, recvVec.size(0)):
            resVec[j] = resVec[j].add(recvVec[index])
            index += 1
                    
    # Perform all-gather now....
    for i in range(size-1):
        dst = (rank + 1) % size
        sendVector = getTensorChunk(
            resVec, (rank - i + 1) % size, totalChunks, chunkSize)
        if sparse:
            s1, s2, s3 = send_sparse(sendVector, dst)
        else:
            send_sig = dist.isend(sendVector, dst=dst)
        recvNode = rank - 1 if rank - 1 >= 0 else size - 1
        recvChunk = (recvNode - i + 1) % size
        sizeList = list(resVec.size())

        if recvChunk == totalChunks - 1:
            sizeList[0] = tensorSize - recvChunk*chunkSize
        else:
            sizeList[0] = chunkSize

        if sparse:
            recvVec = recv_sparse(recvNode, tensorDim,
                                  sizeList, sendVector.dtype, s1, s2, s3)
        else:
            recvVec = torch.zeros(sizeList, dtype=sendVector.dtype)
            recv_sig = dist.irecv(recvVec, src=recvNode)

        if not sparse:
            send_sig.wait()
            recv_sig.wait()

        # Replace received chunk in resVec
        for i in range(0, recvVec.size(0)):
            resVec[recvChunk*chunkSize + i] = recvVec[i]

    resVec /= size
    tensor.copy_(resVec)

######################################################
##### END OF RING ALL REDUCE #########################


def _performAllReduce(tensor: torch.Tensor, rank, size, topology, sparse):
    if topology == "rec-double-half":
        recursive_halving_doubling(rank, tensor, size, sparse)
    elif topology == "ring":
        ring_all_reduce(rank, tensor, size, sparse)
    elif topology == "tree":
        tree_all_reduce(rank, tensor, size, sparse)
    elif topology == "butterfly":
        butterfly_all_reduce(rank, tensor, size, sparse)


def performAllReduce(model, rank, size, topology, sparse):
    for param in model.parameters():
        _performAllReduce(param.data, rank, size, topology, sparse)
