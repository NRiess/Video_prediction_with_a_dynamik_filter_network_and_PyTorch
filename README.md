# Video prediction with a Dynamik filter network

This repository contains a reimplementation of the [code](https://github.com/dbbert/dfn) 
from Bert De Brabandere.
Bert De Brabandere reproduced the experiments of [Dynamic Filter Networks]
(https://arxiv.org/pdf/1605.09673v2.pdf), a NIPS 2016 
paper by Bert De Brabandere*, Xu Jia*, Tinne Tuytelaars and Luc Van Gool (* Bert and 
Xu contributed equally).

Unlike the code of Bert De Brabandere, which is based on a now outdated library 
called lasagne, PyTorch is used instead in this repository.

# Principle of prediction with a dynamic filter network
In a traditional convolutional layer, the learned filters stay fixed after training. 
In contrast, the authors of the [paper](https://arxiv.org/pdf/1605.09673v2.pdf) 
introduce a new framework, the Dynamic Filter Network, where filters are generated 
dynamically conditioned on an input. 

In this way future frames of the moving MNIST dataset can be computed:

![mnist prediction](https://i.imgur.com/XbyD2ix.png)

<!---
![mnist gif1](https://i.imgur.com/vmkSn0k.gif)
![mnist gif2](https://i.imgur.com/JzGhE31.gif)
-->

In this repository the Highway driving dataset is used.


## Create environment

The code has been tested with Anaconda (Python 3.7.12), PyTorch 1.10.0 and CUDA 
11.4 (Different Pytorch + CUDA version is also compatible).  
Please run the provided conda environment setup file:

  ```Shell
  conda env create -f environment.yml
  conda activate dfn-pytorch
  ```

## Install dataset

Please download the download the 
[Highway driving datasets](https://drive.google.com/file/d/1p7K9FBjQBwAbH4UOdLYy2FFBBDsTpF2g/view?usp=sharing) 
and update the paths in the datasets/dataset_highwayDriving.py to point to them.

## Training

Training can be started by running main_pytorch.py.
You can choose the number of frames considered for training with ```--num_of_train_frames```. Please set ```--num_of_train_frames=15693``` to consider the whole training set.
