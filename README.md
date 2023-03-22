** Source code and evaluation scripts for our ECML PKDD 2022 paper. **

[Link to paper](https://link.springer.com/chapter/10.1007/978-3-031-26409-2_27)<br>
[Direct PDF](https://link.springer.com/content/pdf/10.1007/978-3-031-26409-2_27.pdf?pdf=inline%20link)<br>
[arXiv](https://arxiv.org/abs/2208.03641)

## No More Strided Convolutions or Pooling: A New CNN Building Block for Low-Resolution Images and Small Objects

### Abstract

Convolutional neural networks (CNNs) have made resounding success in many computer vision tasks such as image classification and object detection. However, their performance degrades rapidly on tougher tasks where images are of low resolution or objects are small. In this paper, we point out that this roots in a defective yet common design in existing CNN architectures, namely the use of *strided convolution* and/or *pooling layers*, which results in a loss of fine-grained information and learning of less effective feature representations. To this end, we propose a new CNN building block called *SPD-Conv* in place of each strided convolution layer and each pooling layer (thus eliminates them altogether). SPD-Conv is comprised of a *space-to-depth* (SPD) layer followed by a *non-strided* convolution (Conv) layer, and can be applied in most if not all CNN architectures. We explain this new design under two most representative computer vision tasks: object detection and image classification. We then create new CNN architectures by applying SPD-Conv to YOLOv5 and ResNet, and empirically show that our approach significantly outperforms state-of-the-art deep learning models, especially on tougher tasks with low-resolution images and small objects.

### Citation

```
@inproceedings{spd-conv2022,
  title={No More Strided Convolutions or Pooling: A New CNN Building Block for Low-Resolution Images and Small Objects},
  author={Raja Sunkara and Tie Luo},
  booktitle={European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD)},
  year={2022},
}
```

<!---
<embed src="./images/yolov5-spd_final.pdf" type="application/pdf">
-->

### SPD-Conv Building Block:



![losses](https://github.com/pavanraja753/pictures/blob/22d14c2045d99a239a0e544a08de93357b9816cb/spd.png?raw=true)


<!---
### YOLOv5-SPD Architecture


![yolov5-spd](https://github.com/pavanraja753/pictures/blob/main/yolov5-spd_final.png?raw=true)
-->





### Installation

```
# Download the code
git clone https://github.com/LabSAINT/SPD-Conv

# Create an environment
cd SPD-Conv
conda create -n myenv python==3.7
conda activate myenv
pip3 install -r requirements.txt
```

SPD-Conv is evaluated using two most representative computer vision tasks, object detection and image classification. Specifically, we construct YOLOv5-SPD, ResNet18-SPD and ResNet50-SPD, and evaluate them on COCO-2017, Tiny ImageNet, and CIFAR-10 datasets in comparison with several state-of-the-art deep learning models.

### YOLOV5-SPD

```
cd YOLOv5-SPD
```


##### Pre-trained models

The table below gives an overview of the results of our models


| $$\textbf{Model}$$ | $$\textbf{AP}$$ | $$\textbf{AP}_\textbf{S}$$ |  $$\textbf{Params (M)}$$ | $$\textbf{Latency (ms)}$$ |
|----	|:-:|:-:|:-:|:-:|
|  [YOLOv5-spd-n](https://drive.google.com/drive/folders/1RqI5JELROohhxRen78W3hG6N9MMRD-6K?usp=sharing) |  31.0 | 16.0 | 2.2   | 7.3|
|  [YOLOv5-spd-s](https://drive.google.com/drive/folders/1RqI5JELROohhxRen78W3hG6N9MMRD-6K?usp=sharing) | 40.0 | 23.5 | 8.7 |  7.3  |
|  [YOLOv5-spd-m](https://drive.google.com/drive/folders/1RqI5JELROohhxRen78W3hG6N9MMRD-6K?usp=sharing) | 46.5|30.3|24.6|8.4
|  [YOLOv5-spd-l](https://drive.google.com/drive/folders/1RqI5JELROohhxRen78W3hG6N9MMRD-6K?usp=sharing) | 48.5|32.4|52.7|10.3




##### Evaluation

The script `val.py` can be used to evaluate the pre-trained models

```
  $ python val.py --weights './weights/nano_best.pt' --img 640 --iou 0.65 --half --batch-size 1 --data data/coco.yaml
  $ python val.py --weights './weights/small_best.pt' --img 640 --iou 0.65 --half --batch-size 1 --data data/coco.yaml
  $ python val.py --weights './weights/medium_best.pt' --img 640 --iou 0.65 --half --batch-size 1 --data data/coco.yaml 
  $ python val.py --weights './weights/large_best.pt' --img 640 --iou 0.65 --half --batch-size 1 --data data/coco.yaml
  
```

##### Training 


```
python3 train.py --data coco.yaml --cfg ./models/space_depth_n.yaml --weights '' --batch-size 128 --epochs 300 --sync-bn --project space_depth --name space_depth_n

python3 train.py --data coco.yaml --cfg ./models/space_depth_s.yaml --weights '' --batch-size 128 --epochs 300 --sync-bn --project space_depth --name space_depth_s

python3 train.py --data coco.yaml --cfg ./models/space_depth_m.yaml --weights '' --batch-size 32 --epochs 200 --sync-bn --project space_depth --name space_depth_m

python3 train.py --data coco.yaml --cfg ./models/space_depth_l.yaml --hyp hyp.scratch_large.yaml --weights '' --batch-size 20 --epochs 200 --sync-bn --project space_depth --name space_depth_l
```
 


### ResNet18-SPD

ResNet18-SPD model is evaluated on the TinyImageNet dataset

```bash
cd ./../ResNet18-SPD
```

| $\textbf{Model}$ | $\textbf{Dataset}$ | $\textbf{Top-1 accuracy}$ (\%) |
|----|:-:|:-:|
|  [ResNet18-SPD](https://drive.google.com/drive/folders/1RqI5JELROohhxRen78W3hG6N9MMRD-6K?usp=sharing) |  TinyImageNet | 64.52|
|  [ResNet50-SPD](https://drive.google.com/drive/folders/1RqI5JELROohhxRen78W3hG6N9MMRD-6K?usp=sharing) | CIFAR-10| 95.03 |


##### Dataset

Tiny-ImageNet-200 dataset can be downloaded from this link [tiny-imagenet-200.zip](https://drive.google.com/file/d/1xLcRyy7-jLV-ywaGwCHxymX9D05X0g5i/view?usp=sharing)

##### Evaluation

```bash
$ python3 test.py -net resnet18_spd -weights ./weights/resnet18_spd.pt
```

##### Training

```bash
python3 train_tiny.py -net resnet18_spd -b 256 -lr 0.01793 -momentum 0.9447 -weight_decay 0.002113 -gpu -project SPD -name resnet18_spd
```



### ResNet50-SPD

ResNet50-SPD model is implemented on the CIFAR-10 dataset

```bash
cd ./../ResNet50-SPD
```

##### Dataset

```bash
CIFAR-10 dataset will be downloaded automatically by the script
```

##### Evaluation

```bash
# Evaluating resnet50-SPD model
python test.py -weights ./weights/resnet50_spd.pth -net resnet50_spd
```

##### Training


```bash
# Training resnet50-SPD model
$ python3 train.py -net resnet50_spd -gpu
```




