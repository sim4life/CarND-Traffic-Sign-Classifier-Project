#**Traffic Sign Recognition**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./writeup_data/viz.png "Visualization"
[image2a]: ./writeup_data/raw_img.png "Raw image"
[image2b]: ./writeup_data/preproc_img.png "Grayscaling & Preprocessing"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_images/Bumpy_Road.jpg "Traffic Sign 1"
[image5]: ./test_images/Deer_Crossing.jpg "Traffic Sign 2"
[image6]: ./test_images/Give_Way.jpg "Traffic Sign 3"
[image7]: ./test_images/No_Entry.jpg "Traffic Sign 4"
[image8]: ./test_images/Road_Work.jpg "Traffic Sign 5"
[image9]: ./writeup_data/img1_softmax_prob.png "Image1 softmax probability"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/sim4life/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the third code cell of the IPython notebook.  

I used the python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the fourth code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing the frequency of the data with respect to its class.

![Input data visualization][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth code cell of the IPython notebook.

As a first step, I decided to _normalize_ the images to have values within the range (-1, 1). This reduced the accuracy by about 2% but it improved prediction on newly acquired images. Afterwards, I _equalized_ the intensity of each image using the skimage.exposure library rescale_intensity function. I did this processing to ensure that the images have normalized intensity across their pixels.
After that I did grayscaling to convert the image to grayscale using convert the images to grayscale because colors were just adding extra layers of depth without any significant increase in accuracy.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2a] ![alt text][image2b]



####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the second code cell of the IPython notebook.  

My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.

I did not augment data set with additional data in this version but I may try it again with generating more data. This caused loss in training accuracy.



####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 6th, 7th and 8th cell of the ipython notebook.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|	Activation layer											|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|	Activation layer											|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten data	      	| outputs 400 				|
| Fully connected		| Outputs 120        									|
| RELU					|	Activation layer											|
| Dropout					|	Dropout rate 0.75											|
| Fully connected		| Outputs 84        									|
| RELU					|	Activation layer											|
| Dropout					|	Dropout rate 0.75											|
| Fully connected		| Outputs 43        									|


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 9th, 10th, 11th and 12th cell of the ipython notebook.

To train the model, I used _15 epochs_, and a _batch size of 128_. I found out that accuracy didn't improve after 15 epochs and that lower batch size gave better accuracy. I also used a _dropout ratio of 75% keep_ to find optimal results. I used a decaying learning rate that started with 0.001 and got the first drop after 8th epoch to _1/(epoch raise to power Euler's_constant)_, then after 12th epoch to _1/10x(epoch raise to power Euler's_constant)_, then after 14th epoch to _1/20x(epoch raise to power Euler's_constant)_.
I used _Adam Optimizer_ as it was giving far better accuracy than Stochastic Gradient Descent.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 15th and 16th cell of the Ipython notebook.

My final model results were:  
* training set accuracy of 99.6%  
* validation set accuracy of 94.7%  
* test set accuracy of 92.4%

I tried two architectures: first as a ConvNet with Conv5x5 --> Conv1x1 layers, and two fully connected layers; second as a classic LeNet with adjustments. I found out the LeNet performed better training without overfitting and under-fitting. I used other techniques to spice up the training accuracy like adding dropout after every convolution layer and fully connected layer, decay in dropout retain rate, and various combinations of preprocessing the input images.  
I tuned hyper paramers: epochs, batch size, and learning rate decay function. I found out that an adjusted decaying learning rate function with 15 epochs and a batch size of 128 was optimal with high validation accuracy.  
I chose Adam Optimizer as Stochastic Gradient Descent was giving very low validation accuracy even with dropout layers. I chose a constant value of dropout as a decaying value did not help with validation accuracy either. Adding dropout layer after convolution layer also reduced validation accuracy so I kept dropout layer after only fully connected layers. Constant learning rate didn't increase the validation accuracy and caused drop in validation accuracy near the end of epochs.  
I chose classic LeNet with adjustments discussed earlier. It worked fine for the training and validation accuracy with more than 93% validation accuracy in most of the cases. I figured out that a test accuracy of higher than 90% is good for majority of situations. However, to increase the accuracy further, I had to augment the training set with newly generated images. I may attempt augmenting images technique in another revision of the assignment.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because it doesn't have many distinct features to help the trained model differentiate it from other image classes with similar features. Image 2, 3 and 4 should be easy to identify. Image 5 can also be confused with similar image classes the model is trained on.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 20th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Bumpy Road      		| Bicycles crossing   									|
| Wild Animals crossing     			| Wild Animals crossing 										|
| Yield					| Yield											|
| No entry	      		| No entry					 				|
| Road work			| Road work      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 90%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 21st and 22nd cells of the Ipython notebook.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .48         			| Bicycles crossing   									|
| .99     				| Wild Animals crossing 										|
| .99					| Yield											|
| .97	      			| No entry					 				|
| .99				    | Road work      				 			|

For the first image, the top five soft max probabilities were:

![alt text][image9]

The trained network is pretty confused and the correct prediction is 5th in the top 5 probabilities with almost 3% probability. The correct prediction and the current top prediction have similar features adding to the confusion. This can be improved with adding random noise and augmenting input data.

For the rest of the images the model is quite sure about the signs (with probabilities higher than of 0.95), and correctly identified the signs.
