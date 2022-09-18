#!/bin/bash
#BSUB -n 2
#BSUB -q general
#BSUB -G compute-holy
#BSUB -J 'levelset_LS[1-6]'
#BSUB -gpu "num=1:gmodel=TeslaV100_SXM2_32GB:mode=exclusive_process"
#BSUB -R 'gpuhost'
#BSUB -R 'select[mem>16G]'
#BSUB -R 'rusage[mem=16GB]'
#BSUB -M 16G
#BSUB -u binxu.wang@wustl.edu
#BSUB -o  /scratch1/fs1/holy/levelset_LS.%J.%I
#BSUB -a 'docker(pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.9)'

echo "$LSB_JOBINDEX"

export TORCH_HOME="/scratch1/fs1/holy/torch"
#export LSF_DOCKER_SHM_SIZE=16g
#export LSF_DOCKER_VOLUMES="$HOME:$HOME $SCRATCH1:$SCRATCH1 $STORAGE1:$STORAGE1"
param_list='--units resnet50_linf8 .layer4.Bottleneck2 30  3  3  --chan_rng 20 30 
--units resnet50_linf8 .layer4.Bottleneck0 30  3  3  --chan_rng 20 30 
--units resnet50_linf8 .layer3.Bottleneck5 30  7  7  --chan_rng 20 30 
--units resnet50_linf8 .layer3.Bottleneck2 30  7  7  --chan_rng 20 30 
--units resnet50_linf8 .layer2.Bottleneck3 30  13  13  --chan_rng 20 30 
--units resnet50_linf8 .layer1.Bottleneck2 30  28  28  --chan_rng 20 30 
--units resnet50_linf8 .layer4.Bottleneck2 30  3  3  --chan_rng 10 20
--units resnet50_linf8 .layer4.Bottleneck0 30  3  3  --chan_rng 10 20
--units resnet50_linf8 .layer3.Bottleneck5 30  7  7  --chan_rng 10 20
--units resnet50_linf8 .layer3.Bottleneck2 30  7  7  --chan_rng 10 20
--units resnet50_linf8 .layer2.Bottleneck3 30  13  13  --chan_rng 10 20
--units resnet50_linf8 .layer1.Bottleneck2 30  28  28  --chan_rng 10 20
'

export unit_name="$(echo "$param_list" | head -n $LSB_JOBINDEX | tail -1)"
echo "$unit_name"
# Append the extra command to the script.
cd ~/Tuning-Manifold-Level-Sets
python large_scale_proto_ris_cluster.py  $unit_name

#!/usr/bin/env bash