<p align="center">
<img width="1000" height="600" src=https://github.com/BiomedicalMachineLearning/Spacell/blob/master/figure/logo.png>

## Introduction to SpaCell

* **SpaCell** has been developed for analysing spatial transcriptomics (ST) data, which include imaging data of tissue sections and RNA expression data across the tissue sections. The ST data add a novel spatial dimension to the traditional gene expression data, which derive from dissociated cells. The ST data also add molecular information to a typical histological image. Spacell is desinged to integrates the two histopathological imaging and sequencing fields, with the ultimate aim to discover novel biology and to improve histopathological diagnosis.  

* **SpaCell** implements (deep) neural network (NN) models likea multi-input and multi-output autoencoder, transfer learning with or without fine tuning and residual and separable convolutional NN architectures to identify cell types or to predict disease stages. The NN integrates millions of pixel intensity values with thousands of gene expression measurements from spatially-barcoded spots in a tissue. Prior to model training, SpaCell enables users for implement a comprehensive data preprocessing workflow to filter, combine, and normalise images and gene expression matrices. 

## Installation

1. Requirements:  

```
[python 3.6+]
[TensorFlow 1.4.0]
[scikit-learn 0.18]
[keras 2.2.4]
[seaborn 0.9.0]
[opencv 4.1.1]
[pandas 0.25.0]
[pillow 6.1.0]
[python-spams 2.6.1]
[staintools 2.1.2]
...
```
2. Installation:    

2.1 Build from sources

To meet the requirements, we recommend user to use either (1) [conda](https://docs.conda.io/projects/conda/en/latest/index.html) environment:

```
# Download SapCell from GitHub and install all required packages:
git clone https://github.com/BiomedicalMachineLearning/Spacell.git
cd Spacell
conda env create -f requirements.yml
```

```
# To activate environment:
conda activate SpaCell
```

```
# To exit environment:
conda deactivate
```

Or (2) [Docker](https://www.docker.com) container:

```
# Downlaod Docker image
docker pull biomedicalmachinelearning/spacell:latest
```

```
# Run Docker container
docker run \
-it \
-v /path/to/your/data:/home/Spacell/dataset/ \ # mount your local data directory to container
biomedicalmachinelearning/spacell:latest
```

2.2 Install from PyPi  

```pip install SpaCell```

### Build Status
Build Type               | Status                                                                                                                                                                                                                                                                                                                                        | Artifacts
------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------
**Ubuntu Linux 18.04**            | ![Build Status](http://ml-ci.amd.com:21096/job/tensorflow-rocm-release/badge/icon)                                                                                                                                                                        | [Conda](https://docs.conda.io/projects/conda/en/latest/index.html), [Docker](https://www.docker.com), [PyPI](https://pypi.org)
**Centos Linux 7**            | ![Build Status](http://ml-ci.amd.com:21096/job/tensorflow-rocm-release/badge/icon)                                                                                                                                                              | [Conda](https://docs.conda.io/projects/conda/en/latest/index.html), [Docker](https://www.docker.com), [PyPI](https://pypi.org)
**MacOS(Mojave)**            | ![Build Status](http://ml-ci.amd.com:21096/job/tensorflow-rocm-release/badge/icon)                                                                                                                                                                      | [Conda](https://docs.conda.io/projects/conda/en/latest/index.html), [Docker](https://www.docker.com), [PyPI](https://pypi.org)
**MacOS(Catalina)**                | ![Build Status](http://ml-ci.amd.com:21096/job/tensorflow-rocm-release/badge/icon)                                                                                                                                                                  | [Conda](https://docs.conda.io/projects/conda/en/latest/index.html), [Docker](https://www.docker.com), [PyPI](https://pypi.org)
**Windows 10**          | ![Build Status](http://ml-ci.amd.com:21096/job/tensorflow-rocm-release/badge/icon)                                                                                                                                                                    | [Docker](https://www.docker.com), [PyPI](https://pypi.org)

\* `python-spams 2.6.1`, a dependence of `staintools 2.1.2`, is currently not avaliable on _**Windows 10**_ platform. We recommend _**Windows 10**_ user to use Docker container instead of Conda environment. 

## Usage

### Configurations

```config.py```

1. Specify the dataset directory and output directory.
2. Specify model parameters.

### 1. Image Preprocessing

```python image_normalization.py```

### 2. Count Matrix PreProcessing

```python count_matrix_normalization.py```

### 3. Generate paired image and gene count training dataset

```python dataset_management.py```

### 4. Classification

```python spacell_classification.py```

### 5. Clustering

```python spacell_clustering.py -i /path/to/one/image.jpg -l /path/to/iamge/tiles/ -c /path/to/count/matrix/ -e 100 -k 2 -o /path/to/output/```

*  `-e` is number of training epochs
*  `-k` is number of expected clusters

### 6. Clustering Validation and Quantification

```python spacell_validation.py -d /path/to/data -a annotation.png -w wsi.jpeg -m affine_tranformation_matrix.txt -o output_folder -k clustering_predictions.tsv -c annotation_colour_range```

* `-c` is annotation colour range thresholds - blue_low green_low red_low blue_upper green_upper red_low
* `-t` indicates that annotations are not closed paths, so spacell with try to close the paths
* `-f` downscale factor if the input whole slide image has already been downscaled
* `-s` spot size, optional, usually set automatically


## Results

### Classification of ALS disease stages
<p align="center">
<img width="400" height="350" src=https://github.com/BiomedicalMachineLearning/Spacell/blob/master/figure/age_roc_combine.png> 
<img width="400" height="350" src=https://github.com/BiomedicalMachineLearning/Spacell/blob/master/figure/age_confusion_matrix_combine.png> 
 
 
### Clustering for finding prostate cancer region

<p align="center">
<img src=https://github.com/BiomedicalMachineLearning/Spacell/blob/master/figure/clustering_1.png> 

### Clustering for finding inflamed stromal 
 
<p align="center">
<img src=https://github.com/BiomedicalMachineLearning/Spacell/blob/master/figure/clustering_2.png> 

### Clustering for anatomical regions in mouse olfactory bulb (High density ST dataset)

<p align="center">
<img src=https://github.com/BiomedicalMachineLearning/Spacell/blob/master/figure/clustering_3.png> 
 
## Dataset 
For evaluating the algorithm, <a href="https://als-st.nygenome.org">ALS (Amyotrophic lateral sclerosis)</a> dataset, <a href="https://doi.org/10.1038/s41467-018-04724-5">prostate cancer</a> dataset, and a high density spatial transcriptomic <a href="https://www.biorxiv.org/content/10.1101/563338v1">HDST</a> dataset were used.

## Citing Spacell 
If you find Spacell useful in your research, please consider citing:

<a href=" ">Xiao Tan, Andrew T Su, Minh Tran, Quan Nguyen (2019). SpaCell: integrating tissue morphology and spatial gene expression to predict disease cells.</a> (Manuscript is currently under-review)

## The team
The software is under active development by the Biomedical Machine Learning Lab at the Institute for Molecular Bioscience (IMB, University of Queensland).   

Please contact Dr Quan Nguyen (quan.nguyen@uq.edu.au), Andrew Su (a.su@uq.edu.au), and Xiao Tan (xiao.tan@uq.edu.au) for issues, suggestions, and we very welcome collaboration opportunities.
 
