# LSBD-VAE
This is the code for the paper "Quantifying and Learning Symmetry-Based Disentanglement"[1] presented in [ICML2022](https://icml.cc/virtual/2022/spotlight/17702).  Please find a tutorial for running a basic experiment within this repository in this [notebook](notebooks/basic_experiment.ipynb).

## Requirements
This code has been tested with Tensorflow version 2.3.1, this code also requires [Tensorflow Datasets](https://www.tensorflow.org/datasets) and it has been tested with version 3.2.1. 

## Data
The data used in this paper can be downloaded from the following [link](https://drive.google.com/file/d/19JTHk5I5yDnaSq_lX7DKTIvgg6J5Sz3-/view?usp=sharing). Please unzip the files within the main folder of the repository to obtain the same folder structure as detailed. 

## Reproduce Results
In order to reproduce the results obtained by training the LSBD-VAE using semi-supervised training type from the main repository the following:
```console
bash ./experiments/run_paths.py
```
Or to reproduce the results obtained by training with data obtained through random walks over the generative factors type:
```console
bash ./experiments/run_semisup.py
```
The Python file run.py can also be used to train the LSBD-VAE with a specific dataset. Available datasets with toroidal underlying structure are *arrow*, *pixel16*, *modelnet_colors*. And datasets with cylindrical structure are *coil100*, *modelnet40_airplanes".

Type from the main repository folder:
```console
python ./experiments/run.py --dataset NAMEDATASET --epochs 1000
```
## Folder Structure
```
lsbd-vae
│   README.md
│   LICENSE   
└───lsbd_vae
│   └───data_utils
│   |   │   data_loader.py
│   |   │   factor_dataset.py
│   |   │   file_dataset.py
│   |   └───transform_image.py
│   └───metrics
│   |   └───dlsbd_metric.py
|   |
│   └───models
│   |   │   architectures.py
│   |   │   latentspace.py
│   |   │   lsbd_vae.py
│   |   └───reconstruction_losses.py
│   └───testing
|   |   |   test_dlsbd.py
|   |   |   test_lsbd_vae.sh
|   |   └───test_square.sh
│   └───utils
│       │   architectures.py
│       │   latentspace.py
│       │   lsbd_vae.py
│       └───reconstruction_losses.py
└───experiments
|   |   run.py
|   |   run_paths.sh
|   |   run_semisup.sh
│   └───configs
│       └───paper_dataset_parameters.py
└───notebooks
|   └───basic_experiment.ipynb
└───data
    └───arrow_images
    |   └───arrow_64.png
    └───modelnet40
        │   airplane_train.h5
        └───modelnet_color_single_64_64.h5
```


## Contact
For any questions regarding the code refer to [Loek Tonnaer](l.m.a.tonnaer@tue.nl) and [Luis Armando Pérez Rey](l.a.perez.rey@tue.nl)

## Citation
[1] Tonnaer, L., Perez Rey, L.A., Menkovski, V., Holenderski, M., Portegies, J.W. (2022). *Quantifying and Learning Linear Symmetry-Based Disentanglement*. In The Thirty-ninth International Conference on Machine Learning (ICML).
*BibTeX*
```
@article{tonnaer2020quantifying,
  title={Quantifying and Learning Linear Symmetry-Based Disentanglement},
  author={Tonnaer, Loek and Rey, Luis A P{\'e}rez and Menkovski, Vlado and Holenderski, Mike and Portegies, Jacobus W},
  journal={arXiv preprint arXiv:2011.06070},
  year={2020}
}
```
