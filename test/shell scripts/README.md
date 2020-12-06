## Information about the scripts
1. `syn-work.sh`: This scipt will reside in the `home` directory on each node. It triggers the `syntheticSparse.py` file in the `test` directory.
2. `scp-syn-work.sh`: Tiny script to sync the `syn-work.sh` script among the nodes.\
`Usage: ./scp-syn-work.sh 7`\
This will copy `syn-work.sh` from node0 to nodes node1 through node7.
3. `trig-syn.sh`: A script to trigger `syn-work.sh` on other nodes.\
`Usage: ./trig-syn.sh ring 1000,1000 8`\
Where the arguments are `topology, tensor size and world size` respectively. 

## Instructions on how to use the scripts
1. Setup ssh access from node0 to other nodes using the standard `ssh-keygen` method.
2. Test `parallel-ssh` by running `parallel-ssh -i -h followers -O StrictHostKeyChecking=no hostname` where followers file has the list of nodes\
node0\
node1\
node2\
node3\
node4\
node5\
node6\
node7\
and `hostname` is a command that you'd like to execute on each node.
3. Once parallel-ssh is setup go ahead and clone the project repo on all nodes,\
`parallel-ssh -i -h followers -O StrictHostKeyChecking=no 'git clone <repo>'`\
Note: Comment node0 from followers
4. Assuming you've already setup venv on node0, do the same on other nodes using\
`parallel-ssh -i -h followers -O StrictHostKeyChecking=no 'cd CS744-Project; ./envsetup.sh'`
5. Run `scp-syn-work.sh` to copy `syn-work.sh` to each node's home dir.
6. On node0 run `syntheticSparse.py`.
7. On node0 itself in another terminal run `trig-syn.sh` to trigger `syntheticSparse.py` on the other nodes.
