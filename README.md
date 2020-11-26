# CS744-Project
Efficient All Reduce for DDP training
To use:

python main.py --rank <rank> --num-nodes <number of nodes involved> --master-ip <masterip> --topology <ring/rec>


python main.py --rank 0 --num-nodes 4 --master-ip 10.10.1.1 --topology ring


To run horovod, first clone the repository using git submodule update --init --recursive 

It would be better to create a virtualenv at this point, before running the next command. 

Run HOROVOD_WITH_MPI=1 HOROVOD_WITH_PYTORCH=1 pip install -e .[pytorch] on the commandline.

If there are errors, please make sure all dependencies needed for horovod are installed https://horovod.readthedocs.io/en/stable/install_include.html  

To test, run horovodrun -np 4 python examples/pytorch/pytorch_mnist.py 


