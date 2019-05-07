# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[images_vis]: ./output_images/images_vis.jpg "Traffic Signs Visualization"
[labels_vis]: ./output_images/label_frequency.jpg "Labels Frequency Histogram"

[grayscale]: ./output_images/images_grayscale.jpg "Grayscaling"
[noisy]: ./output_images/image_noisy.jpg "Random Noise"
[balanced]: ./output_images/balanced_labels.png "Balanced image data"
[new]: ./output_images/new_images.png "New Images"
[softmax]: ./output_images/softmax_prob.png "Softmax Probabiliy"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/dawn360/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is `len(X_train)` 34799
* The size of the validation set is `len(X_validation)` 4410
* The size of test set is `len(X_test)` 12630
* The shape of a traffic sign image is `X_train[0].shape` (32, 32, 3)
* The number of unique classes/labels in the data set is `len(np.unique(y_train))` 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

![alt text][images_vis]

A bar chart showing how the frequecy of classes in the dataset
![alt text][labels_vis]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because colors are not really useful for training the model. Traffic sign images could be taken with
varying lighting conditions which will affect the color on the traffic sign. our model will be inaccurate trying to guess by color. So hence strip colors use grayscale

Here is an example of a traffic sign image before and after grayscaling.

![alt text][grayscale]

As a last step, I normalized the image data i normalized the image data to improve performance during processing. 

I decided to generate additional data because:

1. the image data distrubution in the test set was not evenly distributed as see in the `label visualization above`. so data was generated for classes with low frequency in the dataset
2. i found that additional data also boosted the accuracy of the model so i generated atleast 1 noisy extra image per test image

To add more data to the the data set, I used the following techniques:

1. Random scaling
2. Random warping
3. Random brightness

Here is an example of an original image and an augmented image:

![alt text][noisy]

The difference between the original data set and the augmented data set is the following:

1. The pespective of the images has slightly changed
2. The images have been scaled
3. also some images have varing brightness

In order to increase the model's accuracy over a few Epoch's i had to expand the training set with a lot of argumented data.
So with a combinatoin of random scaling, brightness and warping i added +100k generated images to the training set.
Size of Trining set after argumentation: 118857 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

|-Layer 1        		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 3x3     	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 	
|-Layer 2
| Input         		| 14x14x6 image   							| 
| Convolution 3x3     	| 1x1 stride, Valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten               | Input = 5x5x16 Output = 400
| Fully connected		| Input = 400 Output = 120
| RELU					|  	
| Fully connected		| Input = 120 Output = 84
| RELU					|     
| Fully connected		| Input = 84 Output = 10
| RELU					| 											|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
After experimenting with several hyperparameters combinations to train my model, i settled with the 
following settings. EPOCHS = 20, BATCH_SIZE = 128 with a learning rate of 0.001 using the AdamOptimizer

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

# LeNet Architecture
![LeNet Architecture](lenet.png)
Source: Yan LeCun

The LeNet architecture was first introduced by LeCun et al. in their 1998 paper, Gradient-Based Learning Applied to Document Recognition. The implementation of LeNet was used primarily for OCR and character recognition in documents.

Using the LeNet Architecture with no modification we are able to achive a Validation accuracy of 95.1 and a test accuracy of 93.6
After several tweeks to the hyperparamenters, we gained a higher accuracy using a higher epoch value with enough augumented image data

EPOCH 1 : Validation Accuracy = 0.851

EPOCH 2 : Validation Accuracy = 0.919

EPOCH 3 : Validation Accuracy = 0.921

EPOCH 4 : Validation Accuracy = 0.934

EPOCH 5 : Validation Accuracy = 0.935

EPOCH 6 : Validation Accuracy = 0.942

EPOCH 7 : Validation Accuracy = 0.941

EPOCH 8 : Validation Accuracy = 0.939

EPOCH 9 : Validation Accuracy = 0.935

EPOCH 10 : Validation Accuracy = 0.944

EPOCH 11 : Validation Accuracy = 0.938

EPOCH 12 : Validation Accuracy = 0.939

EPOCH 13 : Validation Accuracy = 0.945

EPOCH 14 : Validation Accuracy = 0.941

EPOCH 15 : Validation Accuracy = 0.946

EPOCH 16 : Validation Accuracy = 0.938

EPOCH 17 : Validation Accuracy = 0.941

EPOCH 18 : Validation Accuracy = 0.951

EPOCH 19 : Validation Accuracy = 0.950

EPOCH 20 : Validation Accuracy = 0.944

Test Accuracy = 0.936

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][new]
Some of the images might be difficult to classify. eg. the 3rd image has low contrast.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)  | Speed limit (30km/h)   						| 
| Speed limit (20km/h)  | Speed limit (20km/h)   						| 
| Wild Animals Crossing | Wild Animals Crossing					        |
| No passing     		| No passing				 			     	|
| Stop          		| Stop                 							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![alt text][softmax]

The model was very certain in predicting the class for each image with 100% certainty.

