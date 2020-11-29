# CS744-Project
Efficient All Reduce for DDP training

usage: main.py [-h] [--rank RANK] [--master-ip MASTER_IP]
               [--num-nodes NUM_NODES] [--topology TOPOLOGY] [--sparse SPARSE]

Data-parallel training

optional arguments:
  -h, --help            show this help message and exit
  --rank RANK           node rank for distributed training
  --master-ip MASTER_IP
                        master ip
  --num-nodes NUM_NODES
                        total number of nodes
  --topology TOPOLOGY   topology for all reduce: 1) tree 2) butterfly 3) ring
                        4) rec-double-half
  --sparse SPARSE       If true, use sparse representation for reduction

python main.py --rank <rank> --num-nodes <number of nodes involved> --master-ip <masterip> --topology <topology> --sparse <1/0>

eg:
python main.py --rank 0 --num-nodes 4 --master-ip 10.10.1.1 --topology ring --sparse 1


CONSTRAINTS:
For recursive-doubling-halving and butterfly algorithms we only support a power of 2 number of nodes (ex: 4, 8, 16, 32 etc).
Furthermore, for recursive-doubling-halving we only provide support for tensors whose size in dimension 0 is divisible by the number of nodes.
For example: you cannot perform allreduce using this method for tensor of size 9 and 4 nodes. The size must be divisible by 4.


To run horovod, first clone the repository using git submodule update --init --recursive 

It would be better to create a virtualenv at this point, before running the next command. 

Run HOROVOD_WITH_MPI=1 HOROVOD_WITH_PYTORCH=1 pip install -e .[pytorch] on the commandline.

If there are errors, please make sure all dependencies needed for horovod are installed https://horovod.readthedocs.io/en/stable/install_include.html  

To test, run horovodrun -np 4 python examples/pytorch/pytorch_mnist.py 


