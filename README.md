# PaDiM-Anomaly-Detection-Localization-master
This is an implementation of the paper [PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization](https://arxiv.org/pdf/2011.08785).   

This code is heavily borrowed from both SPADE-pytorch(https://github.com/byungjae89/SPADE-pytorch) and MahalanobisAD-pytorch(https://github.com/byungjae89/MahalanobisAD-pytorch) projects.
<p align="center">
    <img src="imgs/pic1.png" width="600"\>
</p>

## Requirement
* python == 3.7
* pytorch == 1.5
* tqdm
* sklearn
* matplotlib

## Datasets
MVTec AD datasets : Download from [MVTec website](https://www.mvtec.com/company/research/datasets/mvtec-ad/)

## Results On ResNet18

### ROC Curve
<p align="center">
    <img src="imgs/roc_curve.png" width="600"\>
</p>

### Localization results
<p align="center">
    <img src="imgs/bottle.png" width="600"\>
</p>
<p align="center">
    <img src="imgs/cable.png" width="600"\>
</p>
<p align="center">
    <img src="imgs/capsule.png" width="600"\>
</p>
<p align="center">
    <img src="imgs/carpet.png" width="600"\>
</p>
<p align="center">
    <img src="imgs/grid.png" width="600"\>
</p>
<p align="center">
    <img src="imgs/hazelnut.png" width="600"\>
</p>
<p align="center">
    <img src="imgs/leather.png" width="600"\>
</p>
<p align="center">
    <img src="imgs/metal_nut.png" width="600"\>
</p>
<p align="center">
    <img src="imgs/pill.png" width="600"\>
</p>
<p align="center">
    <img src="imgs/screw.png" width="600"\>
</p>
<p align="center">
    <img src="imgs/tile.png" width="600"\>
</p>
<p align="center">
    <img src="imgs/toothbrush.png" width="600"\>
</p>
<p align="center">
    <img src="imgs/transistor.png" width="600"\>
</p>
<p align="center">
    <img src="imgs/wood.png" width="600"\>
</p>
<p align="center">
    <img src="imgs/zipper.png" width="600"\>
</p>

## Reference
[1] Thomas Defard, Aleksandr Setkov, Angelique Loesch, Romaric Audigier. *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization*. https://arxiv.org/pdf/2011.08785

[2] https://github.com/byungjae89/SPADE-pytorch

[3] https://github.com/byungjae89/MahalanobisAD-pytorch
