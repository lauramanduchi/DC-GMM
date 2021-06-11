# Deep Conditional Gaussian Mixture Model for Constrained Clustering.

This repository holds the code for Arxiv paper Deep Conditional Gaussian Mixture Model for Constrained Clustering.

## Motivation

Clustering with constraints has gained significant attention in the field of constrained machine learning as it can leverage partial prior information on a growing amount of unlabelled data. 
Following recent advances in deep generative models, we derive a novel probabilistic approach to constrained clustering that can be trained efficiently in the framework of stochastic gradient variational Bayes. 
In contrast to existing approaches, our model (DC-GMM) uncovers the underlying distribution of the data conditioned on prior clustering preferences, expressed as \textit{pairwise constraints}. The inclusion of such constraints allows the user to drive the clustering process towards a desirable configuration by indicating which samples should or should not belong to the same class.

## Data Download

To download Reuters data, run the following:

`cd dataset/reuters`

`sh download_data.sh`

Download STL data (Matlab files) from https://cs.stanford.edu/~acoates/stl10/. Save them in `dataset/stl10/stl10_matlab`. Then run the following:

`cd dataset/stl10`

`python compute_stl_features.py`

To download and configure the UTKFace datset:
- Download the cropped and aligned dataset archive from https://susanqq.github.io/UTKFace/
- Extract the images from this archive to `<code root>/dataset/utkface`

## Implementation

To run DC-GMM using the default setting on MNIST data set:

`python main.py --pretrain True`

To run DC-GMM without pairwise constraints using the default setting:

`python main.py --pretrain True --num_constrains 0`

To choose different configurations of the hyper-parameters:

`python main.py --data ... num_constrains ... --alpha ... --lr ...`

Important hyper-parameters:
- data: choose from MNIST, fMNIST, Reuters, har, utkface
- num_constrains: by default it should be set to 6000 (note that the total number of pairwise constraints in a dataset is O(N*N))
- alpha: measure the confidence in your labels (default is 10000)
- pretrain: False if you want to use your own pretrain weights

### Pairwise constraints 

In the current implementation, the pairwise constraints are obtained from labels by randomly sampled two data points and assigning a must-link constraint (+1) if the two samples have the same label and a cannot-link constraint (-1) otherwise. The pairwise constraints are stored in a matrix W.
See the file:
`source/data.py`
