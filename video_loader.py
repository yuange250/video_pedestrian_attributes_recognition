from __future__ import print_function, absolute_import
import os
from PIL import Image
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset
import random

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None, attr=False, attr_loss="cropy", attr_lens=[], max_seq_len=200, sample_margin=10, dataset_name="mars"):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.attr = attr
        self.attr_loss = attr_loss
        self.attr_lens = attr_lens
        self.max_seq_len = max_seq_len
        self.sample_margin = sample_margin
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid, attrs = self.dataset[index]
        # if self.dataset_name == "mars":
        #     img_paths = list(img_paths)
        #     random.shuffle(img_paths)
        # if self.sample_margin > 1 and len(img_paths) > self.seq_len * self.sample_margin:
        #     if self.sample == 'random':
        #         strat_frame = int(self.sample_margin * random.random())
        #         img_paths = img_paths[strat_frame::self.sample_margin]
        #     elif self.sample == 'dense':
        #         new_dense_img_paths = []
        #         for sp in range(self.sample_margin):
        #             new_dense_img_paths += img_paths[sp::self.sample_margin]
        #         img_paths = new_dense_img_paths
        num = len(img_paths)
        attributes = []
        if self.attr:
            if self.attr_loss == "cropy":
                for a in attrs:
                    attributes.append(Tensor([a]).long())
                if np.sum(attrs[2:9]) == 0:
                    attributes.append(Tensor([1]).long())
                else:
                    attributes.append(Tensor([0]).long())
            # elif self.attr_loss == "mse":
            #     for i, a in enumerate(attrs):
            #         attr = [1 if _ == a else 0 for _ in range(self.attr_lens[i])]
            #         attributes.append(Tensor(attr))
        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = list(range(num))
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]

            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices=np.array(indices)
            imgs = []
            for index in indices:
                index=int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            #imgs=imgs.permute(1,0,2,3)
            return imgs, pid, camid, attributes,

        elif self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            cur_index=0
            frame_indices = list(range(num))
            indices_list=[]
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=frame_indices[cur_index:]
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            indices_list.append(last_seq)
            imgs_list=[]
            for indices in indices_list:
                imgs = []
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)

                imgs = torch.cat(imgs, dim=0)
                #imgs=imgs.permute(1,0,2,3)
                imgs_list.append(imgs)
            if len(imgs_list) > self.max_seq_len:
                sp = int(random.random() * (len(imgs_list) - self.max_seq_len))
                ep = sp + self.max_seq_len
                imgs_list = imgs_list[sp:ep]
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid, attributes, img_paths[0]

        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))







