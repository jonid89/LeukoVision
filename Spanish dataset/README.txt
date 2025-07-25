In this folder, you will find couple of python files to do analysis on the dataset of WBCs images
The first one is to check the quality of the images of the dataset (check_images.py)
In addition, the file train_val_split.py creates a subfolder where the data is split into a training set and validation set.

Inside the subfolder, in the addition to the training and the validation subsets, we proposed three models based on the VGG16 architecture to predict images.
The first model we used the full dataset (all images) and it is saved in h5 file format.
In the second model we used the oversampling technique to ensure the even distribution of classes, this was done by selecting random images and rotating randomly between 1 and 359 degrees until the maximum number of images is reached.
The third model was based on undersampling, in which we used the lowest population as a reference for the other classes.
