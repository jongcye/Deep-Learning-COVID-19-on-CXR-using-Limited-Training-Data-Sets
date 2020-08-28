# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 2020
@author: Yujin Oh (yujin.oh@kaist.ac.kr)
"""

import header

# common
import torch, torchvision
import numpy as np

# dataset
import mydataset
from torch.utils.data import DataLoader
from PIL import Image

# model
import model
import torch.optim as optim
import torch.nn as nn
import os
import glob

# post processing
import cv2


def main():

    print("\ninference.py")


    ##############################################################################################################################
    # Semantic segmentation (inference)

    # Flag  
    flag_eval_JI = False # calculate JI
    flag_save_JPG = False # preprocessed, mask

    # GPU   
    if torch.cuda.is_available():
        device = torch.device("cuda:0") 
        num_worker = header.num_worker 
    else:
        device = torch.device("cpu") 
        num_worker = 0


    # Model initialization
    net = header.net


    # Load model
    model_dir = header.dir_checkpoint + header.filename_model
    if os.path.isfile(model_dir):
        print('\n>> Load model - %s' % (model_dir))
        checkpoint = torch.load(model_dir)
        net.load_state_dict(checkpoint['model_state_dict']) 
        test_sampler = checkpoint['test_sampler']
        print("  >>> Epoch : %d" % (checkpoint['epoch']))
        # print("  >>> JI Best : %.3f" % (checkpoint['ji_best']))
    else:
        print('[Err] Model does not exist in %s' % (header.dir_checkpoint + header.filename_model))
        exit()


    # network to GPU
    net.to(device) 
    

    # loop dataset class
    folder_list = ['Normal', 'COVID-19',  'Virus', 'bacteria', 'TB']
    for folder in folder_list:

        # Dataset
        print('\n>> Load dataset -', header.dir_data_root + folder)
        if flag_eval_JI:
            testset = mydataset.MyTestDataset(header.dir_test_path, test_sampler)   
        else:
            testset = mydataset.MyInferenceClass(tag = folder) 
        testloader = DataLoader(testset, batch_size=header.num_batch_test, shuffle=False, num_workers=num_worker, pin_memory=True)
        print("  >>> Total # of test sampler : %d" % (len(testset)))


        # inference
        print('\n\n>> Evaluate Network')
        with torch.no_grad(): 

            # initialize
            net.eval()
            ji_test = []

            for i, data in enumerate(testloader, 0):
  
                # forward
                outputs = net(data['input'].to(device))
                outputs = torch.argmax(outputs.detach(), dim=1)  

                # one hot
                outputs_max = torch.stack([mydataset.one_hot(outputs[k], header.num_masks) for k in range(len(data['input']))])

                # each case
                for k in range(len(data['input'])): 

                    # get size and case id
                    original_size, dir_case_id, dir_results = mydataset.get_size_id(k, data['im_size'], data['ids'], header.net_label[1:]) 

                    # post processing
                    post_output = [post_processing(outputs_max[k][j].numpy(), original_size) for j in range(1, header.num_masks)] # exclude background
        
                    # jaccard index (JI)
                    if flag_eval_JI:
                        ji = tuple(get_JI(post_output[j], data['masks'][k][j].numpy()) for j in range(len(post_output)))
                        ji_test.append(ji)
                    
                    # original image processings
                    save_dir = header.dir_save + folder
                    mydataset.create_folder(save_dir)
                    image_original = testset.get_original(i*header.num_batch_test+k)
                    np.save(save_dir + dir_case_id + '.image.npy', image_original)
                    np.save(save_dir + dir_case_id + '.mask.npy', post_output[1]+post_output[2])

                    # save mask/pre-processed image
                    if flag_save_JPG:
                        save_dir = save_dir.replace('/' + folder, '_visualize/' + folder)
                        mydataset.create_folder(save_dir)
                        Image.fromarray(post_output[1]*255 + post_output[2]*255).convert('L').save(save_dir + dir_case_id + '_mask.jpg')
                        Image.fromarray(image_original.astype('uint8')).convert('L').save(save_dir + dir_case_id + '_image.jpg')
                    
            # JI statistics
            if flag_eval_JI:
                ji_thorax= [np.mean((ll, rl)) for (heart, ll, rl) in ji_test]
                ji_cardiac = [heart for (heart, ll, rl) in ji_test]


def get_JI(pred_m, gt_m):

    intersection = np.logical_and(gt_m, pred_m)  

    true_sum= gt_m[:,:].sum()
    pred_sum= pred_m[:,:].sum()
    intersection_sum = intersection[:,:].sum()

    ji = (intersection_sum + 1.) / (true_sum + pred_sum - intersection_sum + 1.)

    return ji           


def post_processing(raw_image, original_size, flag_pseudo=0):

    net_input_size = raw_image.shape
    raw_image = raw_image.astype('uint8')

    # resize
    if (flag_pseudo):
        raw_image = cv2.resize(raw_image, original_size, interpolation=cv2.INTER_NEAREST)
    else:
        raw_image = cv2.resize(raw_image, original_size, interpolation=cv2.INTER_NEAREST)    

    if (flag_pseudo):
        raw_image = cv2.resize(raw_image, net_input_size, interpolation=cv2.INTER_NEAREST)

    return raw_image


if __name__=='__main__':

    main()

