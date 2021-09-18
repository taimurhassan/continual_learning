# Incremental Cross-Domain Adaptation for Robust Retinopathy Screening via Bayesian Deep Learning

## Update (September 18th, 2021)
A [supporting document](https://github.com/taimurhassan/continual_learning/blob/main/Supporting_Document.pdf) describing the difference between transfer learning, incremental learning, domain adaptation, and the proposed incremental cross-domain adaptation approach has been uploaded in this repository.

## Update (August 15th, 2021)
[Blind Testing Dataset](https://drive.google.com/file/d/13wFqJ2-uC2CAOqv1rg9fJJcyBD1wJT6w/view?usp=sharing) has been released.

## Introduction
This repository contains an implementation of the continual learning loss function (driven via Bayesian inference) to penalize the deep classification networks for incrementally learning the diverse ranging classification tasks across various domain shifts.

![CL](/images/BD3.png)

## Installation
To run the codebase, please download and install Anaconda (also install MATLAB R2020a with deep learning, image processing and computer vision toolboxes). Afterward, please import the ‘environment.yml’ or alternatively install following packages: 
1. Python 3.7.9 
2. TensorFlow 2.1.0 (CUDA compatible GPU needed for GPU training) 
3. Keras 2.3.0 or above 
4. OpenCV 4.2 
5. Imgaug 0.2.9 or above 
6. Tqdm 
7. Pandas
8. Pillow 8.2.0

Both Linux and Windows OS are supported.

## Datasets
The datasets used in the paper can be downloaded from the following URLs: 

1. [Rabbani](https://sites.google.com/site/hosseinrabbanikhorasgani/datasets-1)
2. [BIOMISA](https://data.mendeley.com/datasets/trghs22fpg/3)
3. [Zhang](https://data.mendeley.com/datasets/rscbjbr9sj/3)
4. [Duke-I](http://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm)
5. [Duke-II](http://people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm)
6. [Duke-III](http://people.duke.edu/~sf59/Srinivasan_BOE_2014_dataset.htm)
7. [**Blind Testing Dataset**](https://drive.google.com/file/d/13wFqJ2-uC2CAOqv1rg9fJJcyBD1wJT6w/view?usp=sharing)

The datasets description file is also uploaded here. Moreover, please follow the same steps as mentioned below to prepare the training and testing data. These steps are also applicable for any custom dataset. Please note that in this research, the disease severity within the scans of all the above-mentioned datasets are marked by multiple expert ophthalmologists. These annotations are also released publicly in this repository.

## Dataset Preparation

1. Download the desired data and put the training images in '…\datasets\trainK' folder (where K indicates the iteration).
2. The directory structure is given below:
```
├── datasets
│   ├── test
│   │   └── test_image_1.png
│   │   └── test_image_2.png
│   │   ...
│   │   └── test_image_n.png
│   ├── train1
│   │   └── train_image_1.png
│   │   └── train_image_2.png
│   │   ...
│   │   └── train_image_m.png
│   ├── train2
│   │   └── train_image_1.png
│   │   └── train_image_2.png
│   │   ...
│   │   └── train_image_j.png
│   ...
│   ├── trainK
│   │   └── train_image_1.png
│   │   └── train_image_2.png
│   │   ...
│   │   └── train_image_o.png
```

## Training and Testing
1. Use ‘trainer.py’ to train the chosen model incrementally. After each iteration, the learned representations are saved in a h5 file.
2. After training the model instances, use ‘tester.py’ to generate the classification results.
3. Use ‘confusionMatrix.m’ to view the obtained results. 

## Results
The detailed results of the proposed framework on all the above-mentioned datasets are stored in the 'results.mat' file. 

## Citation
If you use the proposed scheme (or any part of this code in your research), please cite the following paper:

```
@inproceedings{BayesianIDA,
  title   = {Incremental Cross-Domain Adaptation for Robust Retinopathy Screening via Bayesian Deep Learning},
  author  = {Taimur Hassan and Bilal Hassan and Muhammad Usman Akram and Shahrukh Hashmi and Abdul Hakeem and Naoufel Werghi},
  note = {Submitted in IEEE Transactions on Instrumentation and Measurement},
  year = {2021}
}
```

## Contact
If you have any query, please feel free to contact us at: taimur.hassan@ku.ac.ae.
