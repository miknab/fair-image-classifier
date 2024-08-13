# Towards a fair/unbiased face classifier

This repository contains the code to create an image classifier
using [this data set from Kaggle](https://www.kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv).
The input images show faces of people of different age, ethncities
and gender. The code in this repository can be used to create a 
classifier for either of these three attributes.

The focus of the work in this repository is on potential biases
in the resulting classifiers. Can we build classifiers that perform
equally (or at least very similarly) well on all population subgroups?

To generate the results stored in this repo I used a MacBook Pro (2019) 
with a 2.8 GHz CPU and 16 GB RAM.

## Notes
The implementation of the AlexNet architecture (see facecls/fcmodels.py) makes use of Lambda layers provided by tensorflow.keras. As mentioned [here](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Lambda), this introduces limitations regarding (de)serialization and portability.

## References used in this work:
[1] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. 2012. "ImageNet classification with deep convolutional neural networks." In Proceedings of the 25th International Conference on Neural Information Processing Systems - Volume 1 (NIPS'12). Curran Associates Inc., Red Hook, NY, USA, 1097–1105.

[2] Siddhesh Bangar, Jun 28, 2022, "VGG-Net Architecture Explained", Medium, online, https://medium.com/@siddheshb008/vgg-net-architecture-explained-71179310050f, accessed: Aug 13, 2024

