#**Traffic Sign Recognition** 

##Shangyi Cheng

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

[image1]: ./writeup_image/label_distribution.png "Visualization"
[image2]: ./writeup_image/y_channel.png "Visualization"
[image3]: ./writeup_image/normalization.png "Visualization"
[image4]: ./writeup_image/jitterring.png "Visualization"
[image5]: ./writeup_image/MS_LeNet.png "Visualization"

[image6]: ./writeup_image/test_image_0.jpg "Traffic Sign 0"
[image7]: ./writeup_image/test_image_1.jpg "Traffic Sign 1"
[image8]: ./writeup_image/test_image_2.jpg "Traffic Sign 2"
[image9]: ./writeup_image/test_image_3.jpg "Traffic Sign 3"
[image10]: ./writeup_image/test_image_4.jpg "Traffic Sign 4"

[image11]: ./writeup_image/learning_curve.png "learning"
---

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the NO.6 - NO.7 code cell of the IPython notebook.

As a first step, I decided to convert the images to YUV colar space and only extract Y channel, because Y channel gives enough edge infomation to identify the traffic signs. Besides, less training data can speed up training speed.

Here is an example of a traffic sign image before and after first step.

![alt text][image2]

As a last step, I globally and locally normalized the image data because this can keep the consistancy between traning data and test data as well as strong edge.
Here is an example of a traffic sign image before and after second step.

![alt text][image3]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The train, valid, and test data are already provided in the downloaded dataset. The code for preprocess them is contained in the NO.8 - No.12 cells.

My final training set had 208794 number of images (34799 original images, 173995 jittered images). My validation set and test set had 4410 and 12630 number of images.

The forth and fifth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because more data can cover more traffic signs scenario. To add more data to the the data set, I used the following techniques:

#####1. Random translation of x,y in range [-2, +2] pixels to simulate the different traffic sign position in images

#####2. Random rotation from [-15, 15] degrees simulate the different orientation traffic sign in images

#####3. Random scale from [0.9, 1.1] simulate the different sign size

#####4. Random gussian blur with sigma in range [0,1.5] to simulate motion blur
 
Here is an example of an original image and an augmented image:

![alt text][image4]

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the NO.14 cell of the ipython notebook. The model is a multi-scale LeNet. Cell No.15 is a single scale LeNet which is used for comparasion. We will mainly talk about multi scale LeNet here.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Normalized Y channel image   							| 
| Conv1 5x5   | 1x1 stride, valid padding, outputs 28x28x6 	|
| abs(tanh())					|		activation										|
| Max pooling	     	| 2x2 stride,  outputs 14x14x6 				|
| Conv2 5x5   | 1x1 stride, valid padding, outputs 10x10x16   	|
| Max pooling         | 2x2 stride,  outputs 5x5x16 |
| abs(tanh()) | activation |
| Flatten (fc0_1, fc0_2)   | Flat conv1 and conv2 to get 1176 and 400 output|
| Fully connected(fc1_1, fc1_2)		| fc0_1 -> fc1_1, fc0_2 -> fc1_2 to get 108 features output|       									|
| concatenate	(fc1)| Concatenate fc1_1 and fc1_2 to get fc1 with 216 ouput  	|
|abs(tanh()) | activation|
|	drop			|		75% keep rate						|
|	Fully conncected fc2|	get 100 output|
|abs(tanh()) | activation|
|	drop			|		75% keep rate						|
| Fully connected | get 43 logits |
| softmax| |
|cross_entropy| |
|weighted loss function |   Weight cross entropy bacause the label distribution in training dataset is unbalance|

Here is the model image describ this model:

![alt text][image5]

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the NO.16 - NO.18 cell of the ipython notebook. 

To train the model, I used tf.train.AdamOptimizer with a learning rate 0.0015. The batch size is 512 and 100 epochs.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.9658
* validation set accuracy of 0.9794
* test set accuracy of 0.9605

I started with a typical LeNet from the LeNet lab since this model works quite well for MNIST. With 0.001 learning rate and 128 batch it works quite well with a accuracy about 95% with valid dataset. Then I tested the multi scale LeNet with same parameter setting and it showed better performance so I decided to stick with MS_LeNet model. Then based on experiments, I found the final hyper parameter for learning rate, epochs and batch sizes. However, I didn’t iteratively optimize pooling, convolutional kernel very carefully. When I tested on the testing dataset and web new images, this model shows a test accuracy about 94%. It seems a little bit overfit on this dataset and not working very well on the new web images, so I added a drop out layer to make this model. Besides, since the training dataset label distribution is unbalanced, so I use a weighted loss function to help predict minority traffic signs. 

Following is the learning curve for training:

![alt text][image11]

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]

The second image might be difficult to classify because it has some weird watermark on it which is not exits in training dataset.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No entry   		|  No entry  									| 
| 30 Limit    			| 30 Limit										|
| Road work			| Road Work								|
| Yield      		| Yield	 				|
| Stop		| Stop  							|


The model was able to correctly all the traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 96.05%. The model is quite strong based on the test accuracy and new web test sample is too small. So it is reasonable to have 100% acccuracy.



####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
