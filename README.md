# Metric Learning Application System Development Resource

This repository is a work in progress, currently focusing on __Deep Mmetric Learning__ and contains notebooks for:

- Anomaly detection application example for [MVTec Anomaly Detection dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/)[1], and related small utilities.
- Benchmarks of various deep metric learnings by Anomaly Detection problem setting with MNIST/CIFAR-10 datasets.
- Visualizations such as grad-CAM, failure cases of the benchmarks, and so on.

Possible final goals:

- Test/benchmark resources summary for metric learning methods as well as various applications.
- Utilities and/or library for metric learning applications.
- Links for resources.

## 1. How to run examples

Examples in this repository depend on some external modules/source codes. Follow steps below to install them first.

Then just running each notebook will show you the results.

### 1.1 Install dependent modules first

- Install dl-cliche from github. It is not directly available from pip, install as follows. [dl-cliche](https://github.com/daisukelab/dl-cliche) is a general purpose utility module.

```shell
pip install git+https://github.com/daisukelab/dl-cliche.git@master --upgrade
```

NOTE: You might be asked to install some other modules which dl-cliche depends on.

- Install [fast.ai](https://www.fast.ai/) and other dependent modules.

```shell
pip install fastai
 :
```

1.2 Download external source codes

```shell
wget https://raw.githubusercontent.com/ronghuaiyang/arcface-pytorch/master/models/metrics.py
wget https://raw.githubusercontent.com/KaiyangZhou/pytorch-center-loss/master/center_loss.py
```

## 2. Examples

- `MVTecAD` contains anomaly detection application examples for [MVTec Anomaly Detection dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/)[1].
- `MNIST` contains benchmark examples for MNIST dataset.
- `CIFAR-10` contains benchmark examples for CIFAR-10 dataset.

## 3. Published documents

- MVTec AD - [Medium: Spotting Defects! - Deep Metric Learning Solution For MVTec Anomaly Detection Dataset](https://medium.com/@nizumical/spotting-defects-deep-metric-learning-solution-for-mvtec-anomaly-detection-dataset-c77691beb1eb)
- MVTec AD - (Japanese) [Qiita: 欠陥発見! MVTec異常検知データセットへの深層距離学習(Deep Metric Learning)応用](https://qiita.com/daisukelab/items/e0ff429bd58b2befbb1b)
- CIFAR-10/MNIST - (Japanese) [Qiita: 深層距離学習(Deep Metric Learning)各手法の定量評価 (MNIST/CIFAR10・異常検知)](https://qiita.com/daisukelab/items/5f9ec189959eaf1ef211)

## 4. References

- [1] Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger. MVTec AD - A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection; in: IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2019. https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/mvtec_ad.pdf
