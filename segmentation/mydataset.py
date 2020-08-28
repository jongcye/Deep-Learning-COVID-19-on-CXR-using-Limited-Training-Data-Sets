# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 2020
@author: Yujin Oh (yujin.oh@kaist.ac.kr)
"""

import header 

# common
import torch
from torchvision.transforms import functional as TF
import random
import numpy as np

# dataset
from torch.utils.data import Dataset, DataLoader
import os
import glob
from PIL import Image

# add
import csv
import shutil


class MyTrainDataset(Dataset):

    def __init__(self, image_path, sampler):

        self.sample_arr = glob.glob(str(image_path) + "/*")
        self.sample_arr.sort()
        self.sample_arr = [self.sample_arr[x] for x in sampler]
        self.data_len = len(self.sample_arr)

        self.ids = []
        self.images = []
        self.masks = []

        if not os.path.isdir(image_path):
            raise RuntimeError('Dataset not found or corrupted. DIR : ' + image_path)

        for sample in self.sample_arr:
            self.ids.append(sample.replace(image_path, '').replace('.IMG', ''))
            self.images.append(sample)
            mask_list = []
            for x in header.dir_mask_path:
                mask_candidate = image_path.replace('/JSRT', x) + self.ids[-1] + '.gif'
                if(os.path.isfile(mask_candidate)):
                    mask_list.append(mask_candidate)
            self.masks.append(mask_list)
        

    def __len__(self):
        return self.data_len


    def __getitem__(self, index):

        # images
        images = np.asarray(np.reshape(np.fromfile(self.images[index], dtype='>u2', sep="", ), (header.orig_height, header.orig_width)))
        original_image_size = np.asarray(images.shape)

        # preprocessing
        images = pre_processing(images)
        
        # resize
        images = np.asarray(Image.fromarray(images).resize((header.resize_width, header.resize_height)))
        images = np.ma.masked_greater(images, (1<<header.rescale_bit)-1)
        images = np.ma.filled(images, (1<<header.rescale_bit)-1)
        images = np.ma.masked_less_equal(images, 0)
        images = np.ma.filled(images, 0)

        # masks
        masks = np.asarray([np.asarray(Image.open(x).resize((header.resize_width, header.resize_height))) for x in self.masks[index]])   
                                            
        # mask scaling
        masks = masks/255
        background_mask = np.expand_dims(0.5 * np.ones(masks.shape[-2:]), axis=0)

        # mask lists for networks
        masks_list = []
        masks_cat = np.concatenate((background_mask, masks), axis=0)
        masks_list = np.argmax(masks_cat, axis=0).astype("uint8") 

        return {'input':np.expand_dims(images, 0), 'masks':masks_list.astype("int64"), 'ids':self.ids[index], 'im_size':original_image_size}
        

    def get_id(self, index):

        return self.ids[index]


class MyTestDataset(Dataset):

    def __init__(self, image_path, sampler):

        self.sample_arr = glob.glob(str(image_path) + "/*")
        self.sample_arr.sort()
        self.sample_arr = [self.sample_arr[x] for x in sampler]

        # filter out nodule cases
        sample_nn = []
        for i in self.sample_arr:
            if(i.find('JPCNN')>0):
                sample_nn.append(i)
        self.sample_arr = sample_nn

        self.data_len = len(self.sample_arr)

        self.ids = []
        self.images = []
        self.masks = []
        
        if not os.path.isdir(image_path):
            raise RuntimeError('Dataset not found or corrupted. DIR : ' + image_path)

        for sample in self.sample_arr:
            self.ids.append(sample.replace(image_path, '').replace('.IMG', ''))
            self.images.append(sample)
            mask_list = []
            for x in header.dir_mask_path:
                mask_candidate = image_path.replace('/JSRT', x) + self.ids[-1] + '.gif'
                if(os.path.isfile(mask_candidate)):
                    mask_list.append(mask_candidate)
            self.masks.append(mask_list)


    def __len__(self):
        return self.data_len


    def __getitem__(self, index):

        # images
        images = np.asarray(np.reshape(np.fromfile(self.images[index], dtype='>u2', sep="", ), (header.orig_height, header.orig_width)))
        original_image_size = np.array(images.shape)

        # preprocessing
        images = pre_processing(images)
        images = images.astype('float32')

        # resize
        images = np.asarray(Image.fromarray(images).resize((header.resize_width, header.resize_height))) 

        # mask
        masks = np.asarray([np.asarray(Image.open(x).resize(original_image_size)) for x in self.masks[index]]).astype("int64")    

        # mask scaling
        masks = masks/255

        # mask lists for networks
        masks_list = masks

        return {'input':np.expand_dims(images, 0), 'masks':masks_list.astype("int64"), 'ids':self.ids[index], 'im_size':original_image_size}


    def get_id(self, index):
        return self.ids[index]


    def get_original(self, index):

        images = np.asarray(np.reshape(np.fromfile(self.images[index], dtype='>u2', sep="", ), (header.orig_height, header.orig_width)))
        images = pre_processing(images)

        return images


class MyInferenceClass(Dataset):

    def __init__(self, tag):

        image_path = header.dir_data_root + tag
        self.images = glob.glob(image_path + '/*')
        self.images.sort()
        self.data_len = len(self.images)
        self.ids = []

        if not os.path.isdir(image_path):
            raise RuntimeError('Dataset not found or corrupted. DIR : ' + image_path)

        for sample in self.images:
            self.ids.append(sample.replace(image_path, ''))


    def __len__(self):
        return self.data_len


    def __getitem__(self, index):

        # load and preprocessing
        images = self.get_original(index).astype('float32')
        original_image_size = np.asarray(images.shape)

        # resize
        images = np.asarray(Image.fromarray(images).resize((header.resize_width, header.resize_height)))

        return {'input':np.expand_dims(images, 0), 'ids':self.ids[index], 'im_size':original_image_size}
    

    def get_original(self, index):

        # load image
        images = Image.open(self.images[index])
        if (np.asarray(images).max() <= 255):
            images = images.convert("L")
        images = np.asarray(images)

        # crop blank area - public only
        line_center = images[int(images.shape[0]/2):,int(images.shape[1]/2)]
        if (line_center.min() == 0):
            images = images[:int(images.shape[0]/2)+np.where(line_center==0)[0][0],:]

        # preprocessing
        images = pre_processing(images, flag_jsrt=0)

        return images


def pre_processing(images, flag_jsrt = 10):

    # histogram
    num_out_bit = 1<<header.rescale_bit
    num_bin = images.max()+1

    # histogram specification, gamma correction
    hist, bins = np.histogram(images.flatten(), num_bin, [0, num_bin])
    cdf = hist_specification(hist, num_out_bit, images.min(), num_bin, flag_jsrt)
    images = cdf[images].astype('float32')

    return images


def hist_specification(hist, bit_output, min_roi, max_roi, flag_jsrt):

    cdf = hist.cumsum()
    cdf = np.ma.masked_equal(cdf, 0)

    # hist sum of low & high
    hist_low = np.sum(hist[:min_roi+1]) + flag_jsrt
    hist_high = cdf.max() - np.sum(hist[max_roi:])

    # cdf mask
    cdf_m = np.ma.masked_outside(cdf, hist_low, hist_high)

    # build cdf_modified
    if not (flag_jsrt):
        cdf_m = (cdf_m - cdf_m.min())*(bit_output-1) / (cdf_m.max() - cdf_m.min())
    else:
        cdf_m = (bit_output-1) - (cdf_m - cdf_m.min())*(bit_output-1) / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m.astype('float32'), 0)

    # gamma correction
    cdf = pow(cdf/(bit_output-1), header.gamma) * (bit_output-1)

    return cdf


def one_hot(x, class_count):

    return torch.eye(class_count)[:, x]


def create_folder(directory):

    if not os.path.exists(directory):
        os.makedirs(directory)


def get_size_id(idx, size, case_id, dir_label):

    original_size_w_h = (size[idx][1].item(), size[idx][0].item())
    case_id = case_id[idx]
    dir_results = [case_id + case_id + '_' + j + '.png' for j in dir_label]

    return original_size_w_h, case_id, dir_results


def split_dataset(len_dataset):

    # set parameter
    offset_split_train = int(np.floor(header.train_split * len_dataset))
    offset_split_valid = int(np.floor(header.valid_split * len_dataset))
    indices = list(range(len_dataset))

    # shuffle
    np.random.seed(407)
    np.random.shuffle(indices)

    # set samplers
    train_sampler = indices[:offset_split_train]
    valid_sampler = indices[offset_split_train:offset_split_valid]
    test_sampler = indices[offset_split_train:] #[offset_split_valid:]

    return train_sampler, valid_sampler, test_sampler
