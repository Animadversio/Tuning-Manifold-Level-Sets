# The official code base for "On the Level Sets and Invariance of Neural Tuning Landscapes" in NeurIPS Workshop on Symmetry and Geometry in Neural Representations

This repository contains code to conduct *in silico* experiments for finding elements from the level sets on hidden units, and code to analyze *in vivo* neuronal tuning map data from primates. 

The experimental data is deposited in https://osf.io/gpzm5/ which is from the paper "Tuning Landscapes of the Ventral Stream" in Cell Reports Nov. 2022 .

## Environment Setup
Install `LPIPS` package locally from my fork. 
```bash
git clone https://github.com/Animadversio/PerceptualSimilarity
cd PerceptualSimilarity/
git checkout distmat_dev
pip install -e .
```

```bash
pip install git+https://github.com/Animadversio/PerceptualSimilarity.git@9e7d938b31be8daa76f9c349a0b872b4836f2edd
```

```bash
conda install scikit-image
```
