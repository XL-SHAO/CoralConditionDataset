#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gtb-container_g1.24h
#$ -ac d=nvcr-pytorch-2010,d_shm=64G
#$ -N evaluation_single_model

. ~/net.sh

/home/songjian/.conda/envs/openmmlab/bin/python script/evaluation_single_model.py