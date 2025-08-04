# Notbooks

## Sampling test

Inside the folder `sampling_test`, we proposed three models based on the VGG16 architecture to predict images with different sampling method.
The first model we used the full dataset (all images) and it is saved in h5 file format.
In the second model we used the oversampling technique to ensure the even distribution of classes, this was done by selecting random images and rotating randomly between 1 and 359 degrees until the maximum number of images is reached.
The third model was based on undersampling, in which we used the lowest population as a reference for the other classes.

# CNN Models (03.08.2025)

After checking the result of over and undersampling (the best result is from oversampling)
we proceeded to construct models to predict cell types.
 
 - First:
 we split the data to 70% training, 15% validation and 15% testinig using the the files train_val_test_split.ipynb

 - Second
In the new spilt folder (new_split), we used the file oversampling.ipynb to oversample the training set
by randomaly rotating the the pictures between 1 and 359 degrees and ensuring that the new images have the same resolution.

- Third
We used pre trained models such as InceptionV3, VGG16 and ResNet50 to be trianed on the dataset
the final result of the models is stored in the respective folder.