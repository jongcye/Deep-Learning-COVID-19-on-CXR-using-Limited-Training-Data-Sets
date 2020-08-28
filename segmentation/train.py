# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 2020
@author: Yujin Oh (yujin.oh@kaist.ac.kr)
"""

import header 

# common
import torch
import numpy as np
import copy

# dataset
import mydataset
from torch.utils.data import DataLoader
import glob

# model
import model
import torch.optim as optim
import torch.nn as nn

# time
import time

# evaluate
from inference import get_JI



def main():

    print("\ntrain.py")


    ##############################################################################################################################
    # Semantic segmentation (train)


    # GPU   
    if torch.cuda.is_available():
        device = torch.device("cuda:0") 
        num_worker = header.num_worker
    else:
        device = torch.device("cpu") 
        num_worker = 0

        
    # Model initialization
    print('\n\n>> Generate network - %s' % (header.tag))
    net = header.net


    # Parameter initialization
    ji = np.zeros(header.num_masks)
    ji_best, ji_semi_best = 0, 0
    trained_epoch = 0
    loss_history = []


    # Dataset
    print('\n>> Load data')
    num_dataset = len(glob.glob(str(header.dir_train_path) + "/*"))
    train_sampler, valid_sampler, test_sampler = mydataset.split_dataset(num_dataset)
    trainset = mydataset.MyTrainDataset(header.dir_train_path, train_sampler[:int(header.division_trainset*len(train_sampler))]) 
    valset = mydataset.MyTrainDataset(header.dir_train_path, valid_sampler) 
    trainloader = DataLoader(trainset, batch_size=header.num_batch_train, num_workers=num_worker, shuffle=True, pin_memory=True)
    valloader = DataLoader(valset, batch_size=header.num_batch_test, num_workers=num_worker, shuffle=False, pin_memory=True) 
    print("  >>> Total # of trainset : %d" % (len(trainset)))
    print("  >>> Total # of valset : %d" % (len(valset)))


    # Criterion & optimizer
    weight_label = np.ones(header.num_masks)
    weight_label[0] = header.weight_bk
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weight_label).cuda())
    optimizer = optim.Adam(net.parameters(), lr=header.learning_rate, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 1e-6)


    # network to GPU    
    net.to(device)
    model_state_dict = copy.deepcopy(net.state_dict())
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70, 80, 90], gamma=0.1) 


    ##############################################################################################################################
    ## Train


    # epoch
    for epoch in range(trained_epoch, header.threshold_epoch): 

        # break
        if (epoch >= header.epoch_max):
            break

        # get start time point
        start_time = time.time()


        ##################################################################################################
        # Train with labeled dataset

        # set train status
        print('\n>> Train Network - [%d epoch]' % (epoch+1))
        net.train()

        running_loss = 0

        for i, data in enumerate(trainloader, 0):

            # forward
            outputs = net(data['input'].to(device)) 

            # loss
            loss = criterion(outputs, data['masks'].to(device)) 

            # zero the parameter gradients
            optimizer.zero_grad()

            # loss update
            loss.backward()
            optimizer.step()

            # print statistics
            with torch.no_grad(): 
                running_loss += loss.item()    
            if i % len(trainloader) == (len(trainloader)-1):              
                print('  >>> [%d/%d] Loss : %.3f' % (epoch+1, header.epoch_max, running_loss/(i+1)))
                loss_history.append([running_loss/(i+1), epoch+1])
                    
        # update learning rate
        lr_scheduler.step()


        ##############################################################################################################################
        ## validation
        
        print('\n>> Validate Network [%d epoch]' % (epoch+1))

        # set eval status
        net.eval()
      
        with torch.no_grad(): 
                            
            # initialize
            ji = np.zeros(header.num_masks)
            count_data = 0
            
            for i, data in enumerate(valloader, 0): 

                # forward
                outputs = net(data['input'].to(device)) 
                outputs = torch.argmax(outputs.detach(), dim=1)

                # one hot
                outputs_max = [mydataset.one_hot(outputs[k], header.num_masks) for k in range(len(data['input']))]
                masks_max = [mydataset.one_hot(data['masks'][k], header.num_masks) for k in range(len(data['input']))]    

                # dice score
                for k in range(len(data['input'])):
                    count_data +=  1
                    ji += [get_JI(outputs_max[k][j].numpy(), masks_max[k][j].numpy()) for j in range(header.num_masks)]


            # averaging ji
            ji = ji/(float)(count_data)
            ji_mean = np.mean(ji[1:])

            # print results
            print('  >>> [%d/%d] JI :' % (epoch+1, header.epoch_max), end='')
            [print(' %.3f,' % (ji[x]), end='') for x in range(header.num_masks)]


        ##############################################################################################################################
        ## print duration
        duration = time.time()-start_time
        print('\n>> End of Train & Validation [%d epoch] %d min/epoch, %0.3f sec/mini batch' % (epoch+1, duration/60, duration/(i+1)))
        if (epoch == (trained_epoch)):
            total_time = duration * (header.epoch_max - trained_epoch)
            print('>> Total %d minutes / %d hours expected' % (total_time/ 60, total_time / 3600))


        ##############################################################################################################################
        ## save model
        mydataset.create_folder(header.dir_checkpoint)
        if (ji_mean > ji_best):
            ji_best = ji_mean
            model_state_dict = copy.deepcopy(net.state_dict())
            torch.save({'epoch' : epoch+1,
                        'model_state_dict' : model_state_dict,
                        'ji_best' : ji_best,
                        'ji' : ji,
                        'loss_history' : loss_history,
                        'test_sampler' : test_sampler,
                        }, header.dir_checkpoint + header.filename_model)  
            print('  >>> Saved as %s [%d epoch] - updated' % (header.dir_checkpoint + header.filename_model, epoch+1))    


        ##############################################################################################################################
        ## Final dice     
        print('  >>> JI (Best) : %.3f (%.3f)' % (ji_mean, ji_best))
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')


if __name__=='__main__':
    main()
