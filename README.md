# Introduction 

Fire is very useful discovery of the humanity. Like any other human invention, mis-handeling of fire can cause huge damage to the humanity and nature. Every year, urban fire results huge life and property damage. Similary the tragic loss of natural resources in uncontroleld wildfire is well known. The control of wildfire has becoming a huge challenge by using the treditional technologies. On the other hand, recently, the advancement in technology and especially the machine learning have benifited the society a lot in diverse aspects, ranging from self driving car to cancer research. Hence, it can surely contribute to the technology to early detect the fire so that we can react it earlier before getting it worse and making a lots of damage or being it out of control. We can train a deep neural network to distinguish fire and non-fire situation with very good accuracy. This technology aided with suitable hardware design can be vary useful to minimize the fire hazard. In this work we train a convolutional neural network which can achieve out of sample accuracy of 97% classifying the fire and non-fire images. 




# Data collection method

## Collecting fire images

The secondary source fire image dataset used in this project are primarily from the following links:

https://drive.google.com/file/d/11KBgD_W2yOxhJnUMiyBkBzXDPXhVmvCt/view

https://www.kaggle.com/phylake1337/fire-dataset

https://github.com/cair/Fire-Detection-Image-Dataset/find/master

The first dataset contains 1405 images of fires. There are mostly urban fire and few wildfires. But the data is not cleaned. It contains many images without fire: site before and after the fire, firefighters and their equipment only, map of the area of the fire, a cartoon of fire, etc. So the data cleaning has been done. All black and white images are removed. All images without fire or smoke are removed. All the repeated images had been removed. It was a challenge whether or not to keep the picture involving fire damage and lots of ash. Ash might be the indication that there was a fire. But as a fire detection tool there might be little interest in the post damage picture. So I decided to remove those images as well. After removing all the irrelevant images we retain 768 images in our dataset. In total, I removed 637 images from the dataset. Which is all human labor.

 

The second dataset contains 755 images containing fire and 244 images without fire. Clearly, there is a disparity in the number of images in two categories. Also, the fire images mostly contain wildfire and few urban fires. However, the non-fire images are mostly forest images. A significant amount of fire images are taken at night while the non-fire images are taken on the day. So, the non-fire dataset is not representative. So, I use a non-fire dataset from other sources to make it more representative. Throughout this project JEPG image with RGB, the format is chosen as default. The fire images in this dataset are in .png format. So conversion was necessary to merge into the final dataset. The labels were all correct in this set but there were few duplicate images. After removing all the duplicates there are 717 fire images.

The third dataset contains 110 fire images. No cleaning is done in this dataset other than removing one duplicate. This dataset contains non-fire images as well. But they are mostly indoor images. Almost all images are taken at day. So we have 1595 fire images in total. However, the dataset is not representative of all fire cases. Limitations of the dataset.

There are no indoor fire images.
- There are no road/highway/vehicle fire images.
- So I use web scraping techniques to collect more fire images, which I will discuss shortly. So in total, we have 2000 fire images. 

## Collecting non-fire images

There is a huge collection of images in 8 Scene Categories Dataset which we may use for non-fire images. This can be found in the following link:

https://people.csail.mit.edu/torralba/code/spatialenvelope/

There are 2688 images in various categories: coast, forest, highway, city, mountain, land, street, tall building, etc, and their subcategories. But almost all images are taken at day. This produces the risk, especially for night images with urban artificial light to misclassify as fire images. So, I need another dataset containing night images without fire. So I used night images from the following source:

https://www.visuallocalization.net/datasets/

These are all urban night images. There are 93 such images. So far we have sufficient non-fire images to match the number of fire images. But we have the following limitations

- there are no not sufficient night images.
- Night images are not diverse: forest, highway, etc.


Therefore, I collected more night images without fire. So in total, we have used 698 night-images in this model. This is little over one-third of total non-fire images. 

## Downloading images from the website

Because of the limitations above we decided to collect more data from the website to make the dataset more representative. I used the adobe stock website to collect images. The link is here:

https://stock.adobe.com/

I used the BeautifulSoup package to extract images from the website. I used 43 different keywords to search for fire images and 35 different keywords to search the night images without fire. I cleaned the data removing all the images not containing fire and smoke from the first set. Similarly, I cleaned the second dataset for the non-fire night images. In this way, I added 407 new fire images and 605 new night images.


Here are few example of fire images and non-fire images used in this model. 

TBD 


# Model Architecture

We have 4000 total labeled sample images in total to work with. That means 2000 fire images and 2000 non-fire images. Among them I have used 3000 images for training and 1000 for testing equally splitting among both the labels. Although this dataset is of decent size, it is not enough to train the model from scratch using keras model. And as we see later it is not necessary as well. Instead we can use some of the pretrained model.

I use VGG16 pretrained model. VGG-16 is a trained Convolutional Neural Network (CNN), from Visual Geometry Group (VGG), Department of Engineering Science, University of Oxford. The number 16 means the number of layers with trainable weights. The reference paper is here:

https://arxiv.org/abs/1409.1556

A review article is here:

https://neurohive.io/en/popular-networks/vgg16/

Pretrained model are trained in different data, not necessarily similar to the data we are training in this model. But we can still use it because of the follwing reason: The CNN has series of layers. Each layer learns a set of features from the image data. The lower layers learn fundamental patterns like edeges, lines, curves etc. The higher layers on the other hand are specific to the images on the model. Hence, the featured learned by the lower level can be general to the large class of images, even the images which model did not see during its training. Because of this reason we only use the base of the pre-trained model removing the top. We do this here in two steps.

First, we retain all the base and remove only the dense top layer and train the model. Which gives us validation accuracy close to 90%. And in second step, we unlock the top convolutional model on the base and further train the model. Since we already have a decent accuracy we can imagine that the model is already close to the optimum model. So we only need to fine tune. For this reason we drop the learning rate to 10% of the previous case and train for larger number of iterations. Doing so we achieve a validation accuracy close to 97%. Which is pretty decent result.

## Setting up model

We set the model with VGG16 base and custom top.

Loss: Since this is classification problem and there are two classes, we use the binary cross-entropy as the loss function.

Optimizer: We use RMSprop optimizer with customized learning rate.

Metrics: In addition to the loss we want to observe the accuracy. We optimize our model based on this metric.


## Data generator & data augmentation

For the large dataset it is not convenient to load all the data into memory. So we use image data generator to load the data from hard disc to memory in small batch. We do the same of the training and test set.

Further, when initiating the image data generator we can do the data augmentation. This is the step to create more data from existing data by transforming the image. This artificially provides more data to train. Here we use rotation, translation, shear, zooming and horizontal flip for data augmentation. Other transformations like verticle flip is not suitable. We only do the data augmentation in the training set and not on the validation and test set.

We pass the training data from the train_generator. We train for 30 epochs. We pass the validation data from the validation_generator. We get validation accuracy above to 90% from this.

Here is the graph of training and validation loss and training and validation accuracy. 

TBD


# Fine tuning the model

## Unlocking the top convolutional block

We trained previosuly with only top layer removed from VGG16. Here we unlock top base layer from VGG16 and fine tune the model. Doing so we reduce the learning rate from $10^{-4}$ to $10^{-5}$. We train for the 50 epoches. The model surpass the validation accuracy of 97% shortly after 30 epochs. It is not unlikely to improve the model after 50 epochs. But I am happy with this for now. The future plan is to check with other pre-trained model rather.

The training and validation graphs are here

TBD 


# Error Analysis

In this section we analyze the error of the model, i.e. mis-classified images. We first see few examples of the correctly classified images. Then we visualize the confusion matrix. And finally, we see separately fire images classified as non-fire and non-fire images classified as fire




Demonstrating the classification by the model 

TBD 


Confusion matrix 

TBD 

Observing the misclassified images


TBD 


Some of the misclassified figure have fire but that is too small. So even human observer is easy to confuse with them. Though some of the big explicit fire images are misclassified too. May be that is painting of fire but not the picture. Misclassified fire images are mostly bonfire, stove fire, fire tourch, kitchen fire etc. This is not big surprise because there were not enough fire sample in training set in that categories. 

Looking at this mis-classified set some of the picture actually seem to have fire. So, the problem is about the mis-labeling. Others don't have fire but have artificial red light or are picture with hue of dawn and dusk almost appearing as fire.

Overall the model has done very good job separating those images with solid 97% accuracy in out of sample images.


# Future direction
 
- Although the accuracy is good VGG16 is large and slow to train.

- Try other pre-trained networks: Xception (smaller size higher accuracy), MobileNet (much smaller in size with comparable accuracy). Look here: https://keras.io/api/applications/

- Have test, train and validation split: So far there is only test and validation set.

- The data scraping code is not very general. It collects only 25 images per keyword. It is not generalized for all kinds of website.

- Collecting even more images.








