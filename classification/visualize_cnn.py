# Import modules
from __future__ import print_function
from __future__ import division
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from classification import header
from utils.utils import data_transforms
from utils.utils import initialize_model
from gradcam import GradCAM
from gradcam.utils import visualize_cam
import random
from utils.utils import augmentation
import cv2
import glob

## Detect if we have a GPU available
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Load data
data_dir = header.data_dir

# Model name
model_name = header.model

# Number of classes
num_classes = header.num_classes

# Feature extract
feature_extract = header.feature_extract

# Test epoch
test_epoch = header.inference_epoch

def main():
    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    model_ft.to(device)

    # Temporary header
    # directory - normal, bacteria, TB, COVID-19, virus
    dir_test = '../data/test/COVID/'
    label = 3 # set 3 for COVID-19 for virus class

    # Data loader
    test_imgs = sorted(glob.glob(dir_test + '*.image.npy'))
    test_masks = sorted(glob.glob(dir_test + '*.mask.npy'))

    for img, mask in zip(test_imgs, test_masks):

        test_img = np.load(img)
        test_mask = np.load(mask)

        test_img = Image.fromarray(test_img).resize((1024,1024))
        test_mask = Image.fromarray(test_mask).resize((1024,1024))

        test_img = np.asarray(test_img)
        test_mask = np.round(np.asarray(test_mask))

        test_masked = np.multiply(test_img, test_mask)

        test_normalized = test_masked

        h_whole = test_normalized.shape[0]  # original w
        w_whole = test_normalized.shape[1]  # original h

        background = np.zeros((h_whole, w_whole))
        background_indicer = np.zeros((h_whole, w_whole))

        sum_prob_wt = 0.0

        for i in range(header.repeat):

            non_zero_list = np.nonzero(test_normalized)

            random_index = random.randint(0, len(non_zero_list[0])-1)

            non_zero_row = non_zero_list[0][random_index]  # random non-zero row index
            non_zero_col = non_zero_list[1][random_index]  # random non-zero col index

            X_patch = test_normalized[int(max(0, non_zero_row - (header.img_size / 2))):
                               int(min(h_whole, non_zero_row + (header.img_size / 2))),
                      int(max(0, non_zero_col - (header.img_size / 2))):
                      int(min(w_whole, non_zero_col + (header.img_size / 2)))]

            X_patch_img = data_transforms(augmentation(Image.fromarray(X_patch), rand_p=0.0, mode='test'))
            X_patch_img_ = np.squeeze(np.asarray(X_patch_img))

            X_patch_1 = np.expand_dims(X_patch_img_, axis=0)
            X_patch_2 = np.expand_dims(X_patch_img_, axis=0)
            X_patch_3 = np.expand_dims(X_patch_img_, axis=0)

            X_ = np.concatenate((X_patch_1, X_patch_2, X_patch_3), axis=0)
            X_ = np.expand_dims(X_, axis=0)

            X = torch.from_numpy(X_)
            X = X.to(device)

            checkpoint = torch.load(os.path.join(header.save_dir, str(header.inference_epoch) + '.pth'))
            model_ft.load_state_dict(checkpoint['model_state_dict'])
            model_ft.eval()
            outputs = model_ft(X)
            outputs_prob = F.softmax(outputs)

            prob = outputs_prob[0][label]
            prob_wt = prob.detach().cpu().numpy()

            gradcam = GradCAM.from_config(model_type='resnet', arch=model_ft, layer_name='layer4')

            mask, logit = gradcam(X, class_idx=label)
            mask_np = np.squeeze(mask.detach().cpu().numpy())
            indicer = np.ones((224, 224))

            mask_np = np.asarray(
                cv2.resize(mask_np, dsize=(int(min(w_whole, non_zero_col + (header.img_size / 2)))
                                              - int(max(0, non_zero_col - (header.img_size / 2))),
                                           int(min(h_whole, non_zero_row + (header.img_size / 2)))
                                           - int(max(0, non_zero_row - (header.img_size / 2)))
                                           )))

            indicer = np.asarray(
                cv2.resize(indicer, dsize=(int(min(w_whole, non_zero_col + (header.img_size / 2)))
                                              - int(max(0, non_zero_col - (header.img_size / 2))),
                                           int(min(h_whole, non_zero_row + (header.img_size / 2)))
                                           - int(max(0, non_zero_row - (header.img_size / 2)))
                                           )))


            mask_add = np.zeros((1024, 1024))
            mask_add[int(max(0, non_zero_row - (header.img_size / 2))):
                               int(min(h_whole, non_zero_row + (header.img_size / 2))),
            int(max(0, non_zero_col - (header.img_size / 2))):
                      int(min(w_whole, non_zero_col + (header.img_size / 2)))] = mask_np
            mask_add = mask_add * prob_wt

            indicer_add = np.zeros((1024, 1024))
            indicer_add[int(max(0, non_zero_row - (header.img_size / 2))):
                               int(min(h_whole, non_zero_row + (header.img_size / 2))),
            int(max(0, non_zero_col - (header.img_size / 2))):
                      int(min(w_whole, non_zero_col + (header.img_size / 2)))] = indicer
            indicer_add = indicer_add

            background = background + mask_add
            background_indicer = background_indicer + indicer_add   # number in this indicer means how many time the area included.

            sum_prob_wt = sum_prob_wt + prob_wt

        final_mask = np.divide(background, background_indicer + 1e-7)

        final_mask = np.expand_dims(np.expand_dims(final_mask, axis=0), axis=0)
        torch_final_mask = torch.from_numpy(final_mask)

        test_img = np.asarray(Image.fromarray(test_img).resize((1024, 1024)))
        test_img = (test_img - test_img.min()) / test_img.max()
        test_img = np.expand_dims(test_img, axis=0)
        test_img = np.concatenate((test_img, test_img, test_img), axis=0)
        torch_final_img = torch.from_numpy(np.expand_dims(test_img, axis=0))

        final_cam, cam_result = visualize_cam(torch_final_mask, torch_final_img)

        final_cam = (final_cam - final_cam.min()) / final_cam.max()
        final_cam_np = np.swapaxes(np.swapaxes(np.asarray(final_cam), 0, 2), 0, 1)
        test_img_np = np.swapaxes(np.swapaxes(test_img, 0, 2), 0, 1)

        final_combined = test_img_np + final_cam_np
        final_combined = (final_combined - final_combined.min()) / final_combined.max()

        plt.imshow(final_combined)
        plt.savefig(img.split('.image.npy')[0] +'.patch.heatmap_' + '.png')

if __name__=='__main__':
    main()