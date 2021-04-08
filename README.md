# Latent Bernoulli Autoencoder (LBAE)
A PyTorch implementation of our paper 
[Latent Bernoulli Autoencoder](https://proceedings.icml.cc/static/paper_files/icml/2020/3022-Paper.pdf)
by Jiri Fajtl, Vasileios Argyriou, Dorothy Monekosso and Paolo Remagnino.
This paper was presented at [ICML 2020](https://icml.cc/Conferences/2020).

# Development Environment
All development and experiments were conducted on the following configuration:
* OS: Ubuntu 18.04
* Kernel:  Linux-5.4.0-52-generic-x86_64-with-Ubuntu-18.04-bionic
* Display driver: NVRM version: NVIDIA UNIX x86_64 Kernel Module  435.21  Sun Aug 25 08:17:57 CDT 2019
* CUDA:  10.1.243
* CUDNN:  7603
* GPU: 2xRTX2080Ti

# Dependencies
## Python Packages
* python:  3.6.9
* torch:  1.7.0+cu101
* torchvision:  0.8.1+cu101
* tensorboardX:  2.0
* tensorflow:  1.14.0
* numpy:  1.19.4
* h5py:  2.10.0
* json:  2.0.9
* pickle:  4.0
* imageio:  2.9.0
* sklearn:  0.22.2.post1
* PIL:  8.0.1


## Third Party Code
Here we reference publicly available, third party implementations that are 
included in our code base.

* Precision and recall evaluation code by [Mehdi S. M. Sajjadi](http://msajjadi.com),
[Assessing Generative Models via Precision and Recall](https://arxiv.org/abs/1806.00035) 
This code, updated to work with tensorflow 1.X along with the model weights, is 
available at https://github.com/ok1zjf/precision-recall-distributions 
The original is here https://github.com/msmsajjadi/precision-recall-distributions

* Code to calculate the FID, developed by 
[Institute of Bioinformatics, Johannes Kepler University Linz](http://www.bioinf.jku.at/)
https://github.com/bioinf-jku/TTUR/blob/master/fid.py, is located in https://github.com/ok1zjf/LBAE/fid.py in this repository.

<!-- [Danica J. Sutherland](https://djsutherland.ml/) -->
* Code fragments for the KID calculation were developed by
[Danica J. Sutherland](https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py)
and are included in https://github.com/ok1zjf/LBAE/eval.py in this repository.

* In [sampler.py](https://github.com/ok1zjf/LBAE/sampler.py) we use the 
[Fashia's](https://github.com/fasiha) 
[python port](https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd)
of the John D’Errico’s matlab implementation of the 
[nearestPSD](https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd) 
function based on the [Higham’s 1988 paper](https://doi.org/10.1016/0024-3795(88)90223-6).

* TeePipe class in the [mllogger.py](https://github.com/ok1zjf/LBAE/mllogger.py) 
uses a code fragment from [stackoverflow to direct stdout to a file](https://stackoverflow.com/q/616645).


# Installation
Clone the LBAE repository
```
git clone https://github.com/ok1zjf/LBAE
```

If you want to run the full evaluation, that is you will execute the `eval.sh` script, 
you also need to install the precision & recall package. 
```
git clone https://github.com/ok1zjf/precision-recall-distributions
```
This must be located in the same directory as LBAE e.g.
```
your_projects/LBAE
your_projects/precision-recall-distributions
```

To run the tests bellow, you need to train your own models or download already
trained models. Models for CIFAR-10, CelebA and MNIST experiments can 
be downloaded by running
```
./download.sh model_weights_urls.txt
```

All tests should be run from the experiment directory, for example:
```
cd celeba_bae_res64_ConvResBlock32-qae-z1500-m170-3
```

# Test Reconstruction
Enter the experiment directory and run 
```
python3 ../celeba.py eval
```
Similarly, for CIFAR-10 experiment go to 
```
cd cifar10_bae_res32_ConvResBlock32-qae-z600-m171-3
```
and run
```
python3 ../cifar10.py  eval
```
The corresponding datasets will be downloaded automatically via the
torchvision downloader.

All test images from the test dataset will be reconstructed 
and stored in the experiment directory in subdirectory `reco`.
A mosaic of random 64 reconstructed images will be stored in `reconstructed_eval\sample_0_0.jpg`.

# Test Random Images Generation
Enter the experiment directory and run
```
python3 ../celeba.py eval gen sample_method=cov gen_imgs=1000 
```
The argument `gen_imgs` specifies how many images will be generated.
`sample_method` sets the generative methods and can be:
* `cov`   applies the LBAE method
* `random`  generates images from latents drawn from normal distribution and then binarized
* `int`   performs interpolation between random images from the test dataset

Generated images will be stored in `generated\samples_cov` and an image 
mosaic in `generated\sample_0_cov.jpg` There will be multiple mosaics with increasing
index in the filename depending on the total number of images generated `gen_imgs`. 

# Test Interpolation
For interpolation between `gen_imgs` number of test images over `interpolate_steps` 
interpolation steps run this
```
python3 ../celeba.py eval gen sample_method=int gen_imgs=100 batch_size_test=10 interpolate_steps=10
```
The `batch_size_test` argument specifies number of images processed in a single batch.

Interpolations will be stored in `generated\samples_int` and mosaics, with 
more accessible view of the interpolations, in  `generated\sample_0_int.jpg`.

# Test Attribute Change
In the current version this applies only to CelebA.
To set the CelebA attribute 16 (goatee) to `gen_imgs` number of test images 
and interpolate between the original and modified images run this 
```
python3 ../celeba.py eval gen sample_method=int gen_imgs=100 batch_size_test=10 interpolate_steps=10 set_attr=16
```
As before, the `batch_size_test` argument specifies number of images processed in a single batch and, more importantly,
number of rows rendered in the image mosaic in `generated\sample_0_int_attr_16.0.jpg`

Some example of the CelebA attributes are: 
16 goatee, 15 eyeglasses, 9 blond hair.

# Training
To train a new model create a new configuration in the celeba.py, cifar10.py or mnist.py
or create a new python script for your dataset. The important parameters are:
* `hps.cfg` specifies the configuration name
* `hps.exp_suffix` the experiment id
* `hps.dataset`  dataset name
* `hps.lr[0]` learning rate

Please note that the configuration, as well as other code in this project, originates from
a larger code base that we utilize for several follow up experiments and consequently
there may be some unused configuration parameters. 

Then, in the root of the LBAE directory, run 
```
python3 celeba.py
```

To resume training run the command with argument `l`.
Without any argument the training starts from a random initialization from epoch 0.
```
python3 celeba.py l
```

# Full Evaluation
To measure the reconstruction loss, FID, KID and precision and recall for the 
LBAE and Random generative methods and the random interpolation run this
in the experiment directory.
```
..\eval.sh
```
Please note you need to install the precision-recall-distributions package first.
```
git clone https://github.com/ok1zjf/precision-recall-distributions
```
This must be located in the same directory as LBAE.

The results will be stored in a single text line with space separated values appended to `results.txt` 
in the experiment directory.
The data are formated as name:value pairs, separated by spaces. 
A tag without a value is interpreted as name of the evaluation method with the corresponding results following. 
Each line starts with the evaluation on the test dataset. The fields are:  
```
e:<weights_train_epoch_number> 
loss_eval:<test_reconstruction_loss> 
nref:<number_of_test_reference_images> 
```

Then, the results for individual evaluation follows, labeled by the names of the
directories with the generated images. 
That is, for the generated images it will be `samples_cov` and `samples_random`,
for interpolation `samples_int` and `reco` for reconstruction.
For example, for images generated with the LBAE method there will be this section:
```
samples_cov ntest:<number_of_test_images> fid:<fid_val> kid_mean:<kid_mean_val> kid_std:<kid_std_val>
```

This repeats for all evaluations.
For every `eval.sh` run there will be a new line appended to the `results.txt`

For some evaluations the precision and recall results can follow.
For example, for the evaluation of  images generate with the random method there will be this section:
```
samples_random f8/18:<precision/recall> 
```

## Cite
If you use this code or reference our paper in your work please use the following citation.
```
@inproceedings{fajtl2020latent,
  title={Latent Bernoulli Autoencoder},
  author={Fajtl, Jiri and Argyriou, Vasileios and Monekosso, Dorothy and Remagnino, Paolo},
  booktitle={International Conference on Machine Learning},
  pages={2964--2974},
  year={2020},
  organization={PMLR}
}
```

