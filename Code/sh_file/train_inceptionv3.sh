#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gs-container_g1.24h
#$ -ac d=nvcr-pytorch-2010,d_shm=64G
#$ -N train_inceptionv3

. ~/net.sh

/home/songjian/.conda/envs/openmmlab/bin/python script/train_inceptionv3.py