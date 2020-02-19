import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys, getopt
import PIL
import pydicom
from PIL import Image, ImageOps
import imageio
import scipy.misc
import numpy as np
import glob
import random
#from .augmentations import Compose, RandomRotate, PaddingCenterCrop
from skimage import transform
import numbers

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt
import torch
from torch.utils import data

def find_str(s, char):
    index = 0

    if char in s:
        c = char[0]
        for ch in s:
            if ch == c:
                if s[index:index+len(char)] == char:
                    return index

            index += 1

    return -1

def augment_gamma(data_sample, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, per_channel=False,
                  retain_stats=False):
    if invert_image:
        data_sample = - data_sample
    if not per_channel:
        if retain_stats:
            mn = data_sample.mean()
            sd = data_sample.std()
        if np.random.random() < 0.5 and gamma_range[0] < 1:
            gamma = np.random.uniform(gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
        minm = data_sample.min()
        rnge = data_sample.max() - minm
        data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
        if retain_stats:
            data_sample = data_sample - data_sample.mean() + mn
            data_sample = data_sample / (data_sample.std() + 1e-8) * sd
    else:
        for c in range(data_sample.shape[0]):
            if retain_stats:
                mn = data_sample[c].mean()
                sd = data_sample[c].std()
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data_sample[c].min()
            rnge = data_sample[c].max() - minm
            data_sample[c] = np.power(((data_sample[c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
            if retain_stats:
                data_sample[c] = data_sample[c] - data_sample[c].mean() + mn
                data_sample[c] = data_sample[c] / (data_sample[c].std() + 1e-8) * sd
    if invert_image:
        data_sample = - data_sample
    return data_sample

class PaddingCenterCrop(object):
    def __init__(self, size):
        '''
        Argument crop_size is an integer for square cropping only.

        Performs padding and center cropping to a specified size.
        '''
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        w, h = img.shape[0], img.shape[1]
        th, tw = self.size
        if img.ndim == 2:
            img = Image.fromarray(img, "I")
        else: # assume 3
            img = Image.fromarray(img, "RGB")

        if w >= tw and h >= th:  # crop a center patch
            x1 = int(round((w - tw) / 2.))
            y1 = int(round((h - th) / 2.))
            return np.array(img.crop((x1, y1, x1 + tw, y1 + th)))
        else:  # pad zeros and do center crop
            pad_h = max(th - h, 0)
            pad_w = max(tw - w, 0)
            rem_h = pad_h % 2
            rem_w = pad_w % 2

            # padding lengh
            left = pad_w // 2
            top = pad_h // 2
            right = pad_w // 2 + rem_w
            bottom = pad_h // 2 + rem_h

            border = (left, top, right, bottom)  # left, top, right, bottom
            img = ImageOps.expand(img, border, fill=0)

            # do center crop
            x1 = max(int(round((w - tw) / 2.)), 0)
            y1 = max(int(round((h - th) / 2.)), 0)
            return np.array(img.crop((x1, y1, x1 + tw, y1 + th)))

class Dataloader(data.Dataset):

    def __init__(self,
                 split='train',
                 augmentations=None,
                 k=5,
                 k_split=1,
                 target_size=(256, 256)
                 ):
        self.target_size = target_size
        self.split = split
        self.k = k
        self.split_len = int(39945/self.k)
        self.k_split = int(k_split)
        self.gt_dict = self.load_diagnosis()
        self.augmentations = augmentations
        self.files = self.read_files()
        self.crop_pad = PaddingCenterCrop(target_size)

    def read_files(self):
        d = []
        txt_file = os.path.join('/cluster/home/jessesun/emb_ic/data/dd.txt')
        split_range = list(range((self.k_split-1)*self.split_len, self.k_split*self.split_len))
        with open(txt_file, 'r') as f:
            for i, line in enumerate(f):
                # append as train set
                if self.split == 'train' and i not in split_range:
                    d.append(line)
                
                #append as val set
                elif self.split == 'val' and i in split_range:
                    d.append(line)
                    if i > split_range[-1]: #do not need to iterate through rest of file
                        return d
        return d

    def load_diagnosis(self):
        d = {}
        txt = '/cluster/home/jessesun/emb_ic/data/diagnosis.txt'
        with open(txt, 'r') as f:
            for idx, line in enumerate(f):
                l = line.split(' ')
                key, val = str(l[0]), int(l[1])
                d[key] = val
        return d

    def __len__(self):
        if self.split == "train":
            return 33174 - self.split_len
        else:
            return self.split_len

    def __getitem__(self, i): # i is index
        l = []
        for f in os.listdir(self.files[i][:-1]):
            if f.endswith(".dcm"):
                dcm = pydicom.dcmread(self.files[i][:-1]+"/"+f)
                img_tmp = np.array(dcm.pixel_array).astype(np.int16)
                l.append(img_tmp)
         
        img = torch.zeros(24, 3, self.target_size[0], self.target_size[1])
        ratio = float(max(self.target_size)/max(l[0].shape))
        for x in range(len(l)):
                if x >= 24:
                    break
                scale_vector = [ratio, ratio]
                if l[x].ndim == 3:
                    scale_vector.append(1)
                tmp = transform.rescale(l[x], 
		scale_vector,
                order=1,
                preserve_range=True,
                multichannel=False,
                mode='constant')
                # 0-pad here
                tmp = self.crop_pad(tmp)
                if tmp.min() > 0:
                    tmp -= tmp.min()
                tmp = (tmp-tmp.mean())/tmp.std()
                tmp = self._transform(tmp)
                if tmp.shape[-1] == 3:
                    tmp = tmp.permute(2, 0, 1)
                img[x] = tmp
        if x < 23:
            inf = img.min()
            for i in range(23 - x):
                img[23-i] = torch.ones(3, self.target_size[0], self.target_size[1])*inf

        idx = find_str(self.files[i], "emb")
        edx = idx
        pid = self.files[i][idx:-1]
        c = 0
        for char in pid:
            edx += 1
            if char == '_':
                c += 1
            if c == 2:
                break

        pid = self.files[i][idx: edx-1]
        diagnosis = self.gt_dict[pid]

        data_dict = {
            "id": pid,
            "image": img,
            "gt": diagnosis
        }

        return data_dict

    def _transform(self, img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
            img = np.concatenate((img, img, img), axis=0)
        img = torch.from_numpy(img).float()
        return img

class Dataloader2D(data.Dataset):

    def __init__(self,
                 split='train',
                 augmentations=None,
                 k=5,
                 k_split=1,
                 target_size=(256, 256)
                 ):
        self.target_size = target_size
        self.split = split
        self.k = k
        self.split_len = int(418142/self.k)
        self.k_split = int(k_split)
        self.gt_dict = self.load_diagnosis()
        self.augmentations = augmentations
        self.files = self.read_files()
        self.crop_pad = PaddingCenterCrop(target_size)

    def read_files(self):
        d = []
        txt_file = os.path.join('/cluster/home/jessesun/emb_ic/data/2dd.txt')
        split_range = list(range((self.k_split-1)*self.split_len, self.k_split*self.split_len))
        with open(txt_file, 'r') as f:
            for i, line in enumerate(f):
                # append as train set
                if self.split == 'train' and i not in split_range:
                    d.append(line)

                #append as val set
                elif self.split == 'val' and i in split_range:
                    d.append(line)
                    if i > split_range[-1]: #do not need to iterate through rest of file
                        return d
        return d

    def load_diagnosis(self):
        d = {}
        txt = '/cluster/home/jessesun/emb_ic/data/diagnosis.txt'
        with open(txt, 'r') as f:
            for idx, line in enumerate(f):
                l = line.split(' ')
                key, val = str(l[0]), int(l[1])
                d[key] = val
        return d

    def __len__(self):
        if self.split == "train":
            return 418142 - self.split_len
        else:
            return self.split_len

    def __getitem__(self, i): # i is index
        dcm = pydicom.dcmread(self.files[i][:-1])
        img = np.array(dcm.pixel_array).astype(np.int16)
        ratio = float(max(self.target_size[:1])/max(img.shape))
        scale_vector = [ratio, ratio]
        if img.ndim == 3:
            scale_vector.append(1)
        
        img = transform.rescale(img,
                                scale_vector,
                                order=1,
                                preserve_range=True,
                                multichannel=False,
                                mode='constant')
        # 0-pad here
        img = self.crop_pad(img)
        if img.ndim == 3:
            img = img.transpose(2, 0, 1)
        if img.min() > 0:
            img -= img.min()
        img = (img-img.mean())/img.std()
        img = self._transform(img)

        idx = find_str(self.files[i], "emb")
        edx = idx
        pid = self.files[i][idx:-1]
        c = 0
        for char in pid:
            edx += 1
            if char == '_':
                c += 1
            if c == 2:
                break

        pid = self.files[i][idx: edx-1]
        diagnosis = self.gt_dict[pid]

        #gt = torch.zeros(2, 1)
        #gt[diagnosis] = 1
        data_dict = {
            "id": pid,
            "image": img,
            "gt": diagnosis
        }

        return data_dict

    def _transform(self, img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
            img = np.concatenate((img, img, img), axis=0)
        img = torch.from_numpy(img).float()
        return img

'''
    def mask_to_onehot(self, mask, num_classes=3):
        _mask = [mask == i for i in range(1, num_classes+1)]
        _mask = [np.expand_dims(x, 0) for x in _mask]
        return np.concatenate(_mask, 0)
    
    def onehot_to_binary_edges(self, mask, radius=2, num_classes=3):
        if radius < 0:
            return mask

        # We need to pad the borders for boundary conditions
        mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

        edgemap = np.zeros(mask.shape[1:])

        for i in range(num_classes):
            dist = distance_transform_edt(mask_pad[i, :])+distance_transform_edt(1.0-mask_pad[i, :])
            dist = dist[1:-1, 1:-1]
            dist[dist > radius] = 0
            edgemap += dist
        edgemap = np.expand_dims(edgemap, axis=0)
        edgemap = (edgemap > 0).astype(np.uint8)
        return edgemap
    
    def mask_to_edges(self, mask):
        _edge = mask
        _edge = self.mask_to_onehot(_edge)
        _edge = self.onehot_to_binary_edges(_edge)
        return torch.from_numpy(_edge).float()

    def random_elastic_deformation(self, image, alpha, sigma, mode='nearest',
                                   random_state=None):
        """Elastic deformation of images as described in [Simard2003]_.
    ..  [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.
        """
        assert len(image.shape) == 3

        if random_state is None:
            random_state = np.random.RandomState(None)

        height, width, channels = image.shape

        dx = gaussian_filter(2*random_state.rand(height, width) - 1,
                         sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter(2*random_state.rand(height, width) - 1,
                         sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        indices = (np.repeat(np.ravel(x+dx), channels),
                np.repeat(np.ravel(y+dy), channels),
                np.tile(np.arange(channels), height*width))

        values = map_coordinates(image, indices, order=1, mode=mode)

        return values.reshape((height, width, channels))
'''
if __name__ == '__main__':

    augs = None #Compose([PaddingCenterCrop(352)])
    dataset = Dataloader(split='train', augmentations=augs)
    dloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    for idx, batch in enumerate(dloader):
        img = batch['image']
        print(img.shape, img.max(), img.min())
