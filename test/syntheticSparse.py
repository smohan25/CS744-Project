#!/usr/bin/env python
# coding: utf-8

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist

import argparse

import util

parser = argparse.ArgumentParser(description="Basic Tree AllReduce")
parser.add_argument("--rank", type=int, default=0, help="Rank of the process")
parser.add_argument("--world_size", type=int,default=2, help="No. of processes")
parser.add_argument("--topology", type=str, choices=["tree", "butterfly", 
                    "ring", "rec-double-half"] help="Choose topology to test")

args = parser.parse_args()

# TODO
