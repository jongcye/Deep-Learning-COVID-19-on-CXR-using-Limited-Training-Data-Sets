# Summary
-------
This is the Github repository for "Deep Learning COVID-19 on CXR using Limited Training Data Sets".

![methods](https://user-images.githubusercontent.com/39784965/81655488-3d5de380-9471-11ea-8f4b-b18e5fda7d08.png)

The proposed method consists of two-step approaches, namely "making segmentation mask" and "classification".
Each tasks are uploaded in seperated folders.

# Instructions
-------
To make segmentation masks, you can use codes included in "segmentation folder".
After making segmentation masks, generated masks as well as preprocessed x-ray images should be moved to "classification folder".

The detailed instruction is provided below.

# Segmentation
------
Instructions for segmentation.

# Classification
------
This is detailed instructions for the second step, segmentation.

After creating preprocessed image file (.image.npy) and mask (.mask.npy), seperate and divide these into train, val, and test folders.The name of preprocess image file and mask should be the same, except for the file format.
In our experiments, we randomly divided dataset into 0.7, 0.1, and 0.2 ratio.
Each folder (train, val, test) should contain daughter folder for labels (normal, bacteria, TB, COVID, virus) for dataloader get labels from image folder.

You can access requried codes in "classification" folder.
Before the training, you should specify test name and training options in "header.py".

For training, just run "train.py".

For inference, just run "inference.py".

Finally, to get probabilistic Grad-CAM based saliency map, run "visualize_cnn.py".

# Publication
-------
To cite this, 

@ARTICLE{9090149,
  author={Y. {Oh} and S. {Park} and J. C. {Ye}},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Deep Learning COVID-19 Features on CXR using Limited Training Data Sets}, 
  year={2020},
  volume={},
  number={},
  pages={1-1},}
  
https://ieeexplore.ieee.org/document/9090149
