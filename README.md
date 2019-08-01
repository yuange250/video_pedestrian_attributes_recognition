# video_pedestrian_attributes_recognition
  Codes for the paper "A Temporal Attentive Approach for Video-Based Pedestrian Attribute Recognition".
  
### Introduction
This repository contains PyTorch implementations of temporal modeling methods for video-based pedestrian attributes recognition. It is forked from [Video-Person-ReID](https://github.com/jiyanggao/Video-Person-ReID). Based on that, I implement temporal modeling methods including temporal pooling, temporal attention, RNN and 3D conv. The base loss function and basic training framework remain the same as [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid). **PyTorch 0.3.1, Torchvision 0.2.0 and Python 2.7** is used.

### Motivation
Although previous work proposed many temporal modeling methods and did extensive experiments, but it's still hard for us to have an "apple-to-apple" comparison across these methods. As the image-level feature extractor and loss function are not the same, which have large impact on the final performance. Thus, we want to test the representative methods under an uniform framework.

### Dataset
All experiments are done on MARS, as it is the largest dataset available to date for video-based person reID. Please follow [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) to prepare the data. The instructions are copied here: 

1. Create a directory named `mars/` under `data/`.
2. Download dataset to `data/mars/` from http://www.liangzheng.com.cn/Project/project_mars.html.
3. Extract `bbox_train.zip` and `bbox_test.zip`.
4. Download split information from https://github.com/liangzheng06/MARS-evaluation/tree/master/info and put `info/` in `data/mars` (we want to follow the standard split in [8]). The data structure would look like:
```
mars/
    bbox_test/
    bbox_train/
    info/
```
5. Use `-d mars` when running the training code.

### Usage
To train the model, please run

    python main_video_person_reid.py --arch=resnet50tp
arch could be resnet50tp (Temporal Pooling), resnet50ta (Temporal Attention), resnet50rnn (RNN), resnet503d (3D conv). For 3D conv, I use the design and implementation from [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch), just minor modification is done to fit the network into this person reID system.

In my experiments, I found that learning rate has a significant impact on the final performance. Here are the learning rates I used (may not be the best): 0.0003 for temporal pooling, 0.0003 for temporal attention, 0.0001 for RNN, 0.0001 for 3D conv.

Other detailed settings for different temporal modeling could be found in `models/ResNet.py`
