# video_pedestrian_attributes_recognition
  Codes for the paper "A Temporal Attentive Approach for Video-Based Pedestrian Attribute Recognition".
  
### Introduction
This repository contains PyTorch implementations of temporal modeling methods for video-based pedestrian attributes recognition. It is forked from [Video-Person-ReID](https://github.com/jiyanggao/Video-Person-ReID). Based on that, I implement temporal modeling methods including temporal pooling, temporal attention, RNN and 3D conv for multi-attributes recognition. **PyTorch 0.4.1, Torchvision 0.2.1 and Python 3.7** is used.

### Motivation
Although previous work proposed many temporal modeling methods and did extensive experiments, but it's still hard for us to have an "apple-to-apple" comparison across these methods. As the image-level feature extractor and loss function are not the same, which have large impact on the final performance. Thus, we want to test the representative methods under an uniform framework.

### Dataset

#### Experiments are done on MARS. 
Please follow [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) to prepare the data. The instructions are copied here: 

1. Create a directory named `mars/` under `data/`.
2. Download dataset to `data/mars/` from http://www.liangzheng.com.cn/Project/project_mars.html.
3. Extract `bbox_train.zip` and `bbox_test.zip`.
4. Download split information from https://github.com/liangzheng06/MARS-evaluation/tree/master/info and put `info/` in `data/mars` (we want to follow the standard split in [8]). 
5. Download `mars_attributes.csv` from http://irip.buaa.edu.cn/mars_duke_attributes/index.html, and put the file in `data/mars`. The data structure would look like:
```
mars/
    bbox_test/
    bbox_train/
    info/
    mars_attributes.csv
```
5. Use `-d mars` when running the training code.

#### Experiments are done on Duke.  
1. Create a directory named `duke/` under `data/`.
2. Download dataset to `data/duke/` from http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-VideoReID.zip.
3. Extract `DukeMTMC-VideoReID.zip`.
4. Download `duke_attributes.csv` from http://irip.buaa.edu.cn/mars_duke_attributes/index.html, and put the file in `data/duke`. The data structure would look like:
```
duke/
    train/
    gallery/
    query/
    duke_attributes.csv
```
5. Use `-d duke` when running the training code.

#### Usage
To train the model, please run

    python -u main_video_attr_recog.py --arch=attr_resnet50tp --model_type="ta"
arch could be Temporal Attention Method (--arch=attr_resnet50tp --model_type="ta"), Temporal Pooling Method (--arch=attr_resnet50tp --model_type="tp"), RNN Attention Method (--arch=attr_resnet50tp --model_type="rnn"), 3D conv (--arch=attr_resnet503d). For 3D conv, I use the design and implementation from [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch), just minor modification is done to fit the network into this person attributes recognition system.

Other detailed settings for different temporal modeling could be found in `models/AttrModels.py`
