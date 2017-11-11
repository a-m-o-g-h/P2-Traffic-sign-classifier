#**Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/dataset.png "Visualization"
[image2]: ./new-signs/1x.png "Traffic Sign 1"
[image3]: ./new-signs/2x.png "Traffic Sign 2"
[image4]: ./new-signs/3x.png "Traffic Sign 3"
[image5]: ./new-signs/4x.png "Traffic Sign 4"
[image6]: ./new-signs/5x.png "Traffic Sign 5"
[image7]: ./new-signs/6x.png "Traffic Sign 6"
[image8]: ./new-signs/7x.png "Traffic Sign 7"
[image9]: ./new-signs/8x.png "Traffic Sign 8"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the number of data samples for each unique class

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The dataset was not augmented.Only minimal amount of preprocessing was done.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

Input : 32x32x3
Layer 1: Convolutional. The output shape should be 28x28x16.
Activation
Dropout
Pooling. The output shape should be 14x14x16.
Layer 2: Convolutional. The output shape should be 10x10x42.
Activation
Dropout
Pooling. The output shape should be 5x5x42.
Flatten. Flatten the output shape of the final pooling layer.This has 1050 outputs.
Layer 3: Fully Connected. This have 350 outputs.
Activation
Layer 4: Fully Connected. This have 175 outputs.
Activation
Layer 5: Fully Connected (Logits). This have 43 outputs.

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used adam optimizer.
Batch size = 24
Number of epochs = 30
learning rate = 0.0007
Keep probability of dropout in training =0.95

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of  0.937 
* test set accuracy of 0.928

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
First the Lenet solution was chosen as the basic architecture. It is chosen since it proved good for Digit recognition set which is image classification similar to that of traffic sign classification
* What were some problems with the initial architecture?
The accuracy of prediction was too low on validation set. it was around 0.6
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
The architecture was adjusted based on trail and error. Also 2 dropout layers were added.
* Which parameters were tuned? How were they adjusted and why?
All the parameters were tuned to provide better prediction of the traffic signs
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Dropout layer helps in creating successful model since the model will learn not to entirely depend on only some inputs because based on keep_prob in dropout , the input signal will be dropped.

If a well known architecture was chosen:
* What architecture was chosen?
LeNet
* Why did you believe it would be relevant to the traffic sign application?
It is chosen since it proved good for Digit recognition set which is image classification similar to that of traffic sign classification
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 High accuracy on the test set which was tested only at last proves that model is working well.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 8 German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7] ![alt text][image8] ![alt text][image9]
Each image have different lighting condition and have different angle and position of traffic sign. 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)      		| Priority road	 									| 
| Road work   			        | Go straight or left 							|
| Speed limit (60km/h)			| Vehicles over 3.5 metric tons prohibited							|
| Right-of-way at the next intersection	| Right-of-way at the next intersection                             |         			       | Keep right	      		        | Keep right					 				|
| General caution		        | General caution     							|
| Turn left ahead		        | No Passing    							|
| Priority road		                | Priority road	     							|

The model was able to correctly guess 4 of the 8 traffic signs, which gives an accuracy of 50%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

|                          Probability         	                                         |     Prediction	        		| 
|:--------------------------------------------------------------------------------------:|:---------------------------------------------:| 
     							
|  1.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00 | 12,  0,  1,  2,  3|
|[  5.28140783e-01,   4.71859187e-01,   2.13223839e-08, 5.50148060e-09,   1.69778878e-11] | [37,  4, 31, 26,  8]|
|[  1.00000000e+00,   4.29410907e-09,   1.52628680e-12, 5.76640894e-13,   2.48965643e-13] | [16, 12, 11,  3,  6]|
|[  1.00000000e+00,   4.59796546e-26,   5.72651716e-34, 0.00000000e+00,   0.00000000e+00] | [11, 12, 40,  0,  1]|
|[  1.00000000e+00,   2.13449137e-11,   4.80272181e-16, 4.61473021e-18,   4.61355065e-18] | [38, 18, 11,  8, 20]|
|[  1.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00] | [18,  0,  1,  2,  3]|
|[  1.00000000e+00,   1.20970040e-22,   3.15810050e-23, 8.00450771e-24,   6.41689274e-27] | [ 9, 41, 16, 32,  3]|
|[  1.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00] | [12,  0,  1,  2,  3]|

The numbers represent unique class of traffic sign as follows
| ClassId		        |     SignName	        					| 
|:---------------------:|:---------------------------------------------:| 

|0|Speed limit (20km/h)|
|1|Speed limit (30km/h)|
|2|Speed limit (50km/h)|
|3|Speed limit (60km/h)|
|4|Speed limit (70km/h)|
|5|Speed limit (80km/h)|
|6|End of speed limit (80km/h)|
|7|Speed limit (100km/h)|
|8|Speed limit (120km/h)|
|9|No passing|
|10|No passing for vehicles over 3.5 metric tons|
|11|Right-of-way at the next intersection|
|12|Priority road|
|13|Yield|
|14|Stop|
|15|No vehicles|
|16|Vehicles over 3.5 metric tons prohibited|
|17|No entry|
|18|General caution|
|19|Dangerous curve to the left|
|20|Dangerous curve to the right|
|21|Double curve|
|22|Bumpy road|
|23|Slippery road|
|24|Road narrows on the right|
|25|Road work|
|26|Traffic signals|
|27|Pedestrians|
|28|Children crossing|
|29|Bicycles crossing|
|30|Beware of ice/snow|
|31|Wild animals crossing|
|32|End of all speed and passing limits|
|33|Turn right ahead|
|34|Turn left ahead|
|35|Ahead only|
|36|Go straight or right|
|37|Go straight or left|
|38|Keep right|
|39|Keep left|
|40|Roundabout mandatory|
|41|End of no passing|
|42|End of no passing by vehicles over 3.5 metric tons|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


