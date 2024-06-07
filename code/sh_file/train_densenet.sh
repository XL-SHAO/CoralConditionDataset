#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gtn-container_g8.24h
#$ -ac d=nvcr-pytorch-2010,d_shm=121G
#$ -N train_densenet_201

. ~/net.sh

/home/songjian/.conda/envs/openmmlab/bin/python script/train_densenet.py