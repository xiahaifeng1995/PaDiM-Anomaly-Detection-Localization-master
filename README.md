# PaDiM-Anomaly-Detection-Localization-master
This is an implementation of the paper [PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization](https://arxiv.org/pdf/2011.08785).   

This code is heavily borrowed from both SPADE-pytorch(https://github.com/byungjae89/SPADE-pytorch) and MahalanobisAD-pytorch(https://github.com/byungjae89/MahalanobisAD-pytorch) projects
<p align="center">
    <img src="imgs/pic1.png" width="1000"\>
</p>

## Requirement
* python == 3.7
* pytorch == 1.5
* tqdm
* sklearn
* matplotlib

## Datasets
MVTec AD datasets : Download from [MVTec website](https://www.mvtec.com/company/research/datasets/mvtec-ad/)


## Results
### Implementation results on MVTec
* Image-level anomaly detection accuracy (ROCAUC)

|MvTec|R18-Rd100|WR50-Rd550|
|:---:|:---:|:---:|
|Carpet| 0.984| 0.999|
|Grid|0.898 | 0.957|
|Leather|0.988 | 1.0|
|Tile| 0.959| 0.974|
|Wood|0.990 | 0.988|
|All texture classes| 0.964| 0.984|
|Bottle|0.996 | 0.998|
|Cable| 0.855| 0.922|
|Capsule|0.870 | 0.915|
|Hazelnut|0.841 |0.933 |
|Metal nut| 0.974| 0.992|
|Pill|0.869 | 0.944|
|Screw| 0.745| 0.844|
|Toothbrush|0.947 |0.972 |
|Transistor| 0.925| 0.978|
|Zipper| 0.741| 0.909|
|All object classes|0.876|0.941 |
|All classes| 0.905|0.955 |

* Pixel-level anomaly detection accuracy (ROCAUC)

|MvTec|R18-Rd100|WR50-Rd550|
|:---:|:---:|:---:|
|Carpet| 0.988| 0.990|
|Grid| 0.936| 0.965|
|Leather|0.990 |0.989 |
|Tile|0.917 | 0.939|
|Wood| 0.940| 0.941|
|All texture classes| 0.953|0.965 |
|Bottle|0.981 | 0.982|
|Cable|0.949| 0.968|
|Capsule| 0.982| 0.986|
|Hazelnut|0.979 | 0.979|
|Metal nut| 0.967|0.971 |
|Pill|0.946 |0.961 |
|Screw| 0.972| 0.983|
|Toothbrush|0.986 |0.987 |
|Transistor| 0.968|0.975 |
|Zipper|0.976| 0.984|
|All object classes|0.971|0.978 |
|All classes| 0.965| 0.973|

 ### ROC Curve

* ResNet18

<p align="center">
    <img src="imgs/roc_curve_r18.png" width="1000"\>
</p>

* Wide_ResNet50_2

<p align="center">
    <img src="imgs/roc_curve_wr50.png" width="1000"\>
</p>

### Localization examples

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
