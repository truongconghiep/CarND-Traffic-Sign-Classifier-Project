# **Traffic Sign Recognition** 

Student: Hiep Truong Cong
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

[image1]: ./write_up/OriginalBachartVisualization.jpg "Barchart Visualization"
[image2]: ./write_up/OriginalImages.jpg "Original images"
[image3]: ./write_up/BrightnessAugmentedImages.jpg "Brightness augmented images"
[image4]: ./write_up/GeneratedImages.jpg "Generated images"
[image5]: ./write_up/GrayscaledImages.jpg "Grayscaled images"
[image6]: ./write_up/NormalizedImages.jpg "Normalized images"
[image7]:  ./write_up/AugmentedDataBarchart.jpg "Augmented image data"
[image8]:  ./write_up/TrainingLenetEpoch10.jpg "Training LeNet Epoch = 10"
[image9]:  ./write_up/TrainingModified_LenetEpoch10.jpg "Training Modified_LeNet Epoch = 10"
[image10]:  ./write_up/TrainingModiied_Lenet_1_Epoch10.jpg "Training Modified_LeNet_1 Epoch = 10"
[image11]:  ./write_up/TrainingModiied_Lenet_2Epoch10.jpg "Training Modified_LeNet_2 Epoch = 10"



---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

##### 1. Files Submitted:
* [Jupyter notebook code](https://github.com/truongconghiep/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)
* [HTML file](https://github.com/truongconghiep/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html)
* [Write-up file](https://github.com/truongconghiep/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)
##### 2. Dataset Exploration
* Dataset Summary:
   * Number of examples in each data sub-set (e.g. traing, validation and test data subsets)
   * Image data shape
   * number of classes
* Exploratory Visualization
   * Table visualization
   * Barchart diagram visualization
* Design and Test a Model Architecture
   * Preprocessing: discription of the chosen preprocessing techniques
   * Model Architecture
     + Discription of the optimizer and other parameters
     + Discusion
   * Model Training: discusion of training parameters and training process
   * Solution Approach: parameter tuning process to find the solution
 * Test a Model on New Images
   * Acquiring New Images
     + Load and visualize images
     + Discusion
   * Performance on new Images: performance comparision of accuracy on captured images and on the test set
   * Model Certainty - Softmax Probabilities: discusion about the certainty of the model
### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

* Table visualization

| labels | Training examples | Test exsamples | Validation exsamples |                     Class name                     |
|:------:|:-----------------:|:--------------:|:--------------------:|:--------------------------------------------------:| 
|   0    |        180        |       60       |          30          |                Speed limit (20km/h)                |
|   1    |       1980        |      720       |         240          |                Speed limit (30km/h)                |
|   2    |       2010        |      750       |         240          |                Speed limit (50km/h)                |
|   3    |       1260        |      450       |         150          |                Speed limit (60km/h)                |
|   4    |       1770        |      660       |         210          |                Speed limit (70km/h)                |
|   5    |       1650        |      630       |         210          |                Speed limit (80km/h)                |
|   6    |        360        |      150       |          60          |            End of speed limit (80km/h)             |
|   7    |       1290        |      450       |         150          |               Speed limit (100km/h)                |
|   8    |       1260        |      450       |         150          |               Speed limit (120km/h)                |
|   9    |       1320        |      480       |         150          |                     No passing                     |
|   10   |       1800        |      660       |         210          |    No passing for vehicles over 3.5 metric tons    |
|   11   |       1170        |      420       |         150          |       Right-of-way at the next intersection        |
|   12   |       1890        |      690       |         210          |                   Priority road                    |
|   13   |       1920        |      720       |         240          |                       Yield                        |
|   14   |        690        |      270       |          90          |                        Stop                        |
|   15   |        540        |      210       |          90          |                    No vehicles                     |
|   16   |        360        |      150       |          60          |      Vehicles over 3.5 metric tons prohibited      |
|   17   |        990        |      360       |         120          |                      No entry                      |
|   18   |       1080        |      390       |         120          |                  General caution                   |
|   19   |        180        |       60       |          30          |            Dangerous curve to the left             |
|   20   |        300        |       90       |          60          |            Dangerous curve to the right            |
|   21   |        270        |       90       |          60          |                    Double curve                    |
|   22   |        330        |      120       |          60          |                     Bumpy road                     |
|   23   |        450        |      150       |          60          |                   Slippery road                    |
|   24   |        240        |       90       |          30          |             Road narrows on the right              |
|   25   |       1350        |      480       |         150          |                     Road work                      |
|   26   |        540        |      180       |          60          |                  Traffic signals                   |
|   27   |        210        |       60       |          30          |                    Pedestrians                     |
|   28   |        480        |      150       |          60          |                 Children crossing                  |
|   29   |        240        |       90       |          30          |                 Bicycles crossing                  |
|   30   |        390        |      150       |          60          |                 Beware of ice/snow                 |
|   31   |        690        |      270       |          90          |               Wild animals crossing                |
|   32   |        210        |       60       |          30          |        End of all speed and passing limits         |
|   33   |        599        |      210       |          90          |                  Turn right ahead                  |
|   34   |        360        |      120       |          60          |                  Turn left ahead                   |
|   35   |       1080        |      390       |         120          |                     Ahead only                     |
|   36   |        330        |      120       |          60          |                Go straight or right                |
|   37   |        180        |       60       |          30          |                Go straight or left                 |
|   38   |       1860        |      690       |         210          |                     Keep right                     |
|   39   |        270        |       90       |          30          |                     Keep left                      |
|   40   |        300        |       90       |          60          |                Roundabout mandatory                |
|   41   |        210        |       60       |          30          |                 End of no passing                  |
|   42   |        210        |       90       |          30          | End of no passing by vehicles over 3.5 metric tons |

* Barchart visualization
![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)
* As we can see in the barchart diagram above, numbers of samples of classes vary in a large range from 180 to 2010. With this example distribution the trained model will prone to classes, which have more example. To make the model to behave equally to every class data augumentation is needed. In this project some augmentation techniques are applied: brightness augmentation, generation of additional data, grayscale conversion and normalization. The augmentation techniques are deployed in following order:
  + Brightness augmentation
  + Generate additional data for training
  + Converting images to grayscale
  + Image normalization
* Let's use the image below as our original images to demonstrate the augmentation techniques
![alt text][image2]
* Brightness augmentation. In this step the brightness of data images is changed randomly, in this way the model will be insensitive to brightness changes in the training images, so the model learns to not rely on brightness information. Brightness augmented images are used as input in the next step. Some brightness augmented images are shown below
![alt text][image3]
* Generate additional data for training. In this step additional data images are generated. The augmentation is done via a number of random transformation, so that the model would never see twice the exact same picture. This helps prevent overfitting and helps the model generalize better. In Keras this can be done via the keras.preprocessing.image.ImageDataGenerator class. This class allows to:
  + configure random transformations and normalization operations to be done on your image data during training
  + instantiate generators of augmented image batches (and their labels) via .flow(data, labels) or .flow_from_directory(directory). These generators can then be used with the Keras model methods that accept data generators as inputs, fit_generator, evaluate_generator and predict_generator
  + In this project following transfomations are applied:
    <pre><code>
        datagen = ImageDataGenerator(rotation_range=4,     # rotation transformation
                                 width_shift_range=0.05,     # horizontal shift
                                 height_shift_range=0.05,    # verticl shift
                                 zoom_range=0.05,            # Zoom 
                                shear_range=0.05)            # shearing </code></pre>
   +  Some generated images are shown below
   ![alt text][image4]
* Converting images to grayscale. Grayscaled images are shown below
   ![alt text][image5]
* Image normalization. Normalized images are shown below
   ![alt text][image6]
* Visualization of the augmented data set

| labels | Training | Test | Validation |                                   Class name                                    |
|:------:|:-----------------:|:--------------:|:--------------------:|:--------------------------------------------------:| 
|   0    |   2010   |  60  |     30     |                              Speed limit (20km/h)                               |
|   1    |   2010   | 720  |    240     |                              Speed limit (30km/h)                               |
|   2    |   2010   | 750  |    240     |                              Speed limit (50km/h)                               |
|   3    |   2010   | 450  |    150     |                              Speed limit (60km/h)                               |
|   4    |   2010   | 660  |    210     |                              Speed limit (70km/h)                               |
|   5    |   2010   | 630  |    210     |                              Speed limit (80km/h)                               |
|   6    |   2010   | 150  |     60     |                           End of speed limit (80km/h)                           |
|   7    |   2010   | 450  |    150     |                              Speed limit (100km/h)                              |
|   8    |   2010   | 450  |    150     |                              Speed limit (120km/h)                              |
|   9    |   2010   | 480  |    150     |                                   No passing                                    |
|   10   |   2010   | 660  |    210     |                  No passing for vehicles over 3.5 metric tons                   |
|   11   |   2010   | 420  |    150     |                      Right-of-way at the next intersection                      |
|   12   |   2010   | 690  |    210     |                                  Priority road                                  |
|   13   |   2010   | 720  |    240     |                                      Yield                                      |
|   14   |   2010   | 270  |     90     |                                      Stop                                       |
|   15   |   2010   | 210  |     90     |                                   No vehicles                                   |
|   16   |   2010   | 150  |     60     |                    Vehicles over 3.5 metric tons prohibited                     |
|   17   |   2010   | 360  |    120     |                                    No entry                                     |
|   18   |   2010   | 390  |    120     |                                 General caution                                 |
|   19   |   2010   |  60  |     30     |                           Dangerous curve to the left                           |
|   20   |   2010   |  90  |     60     |                          Dangerous curve to the right                           |
|   21   |   2010   |  90  |     60     |                                  Double curve                                   |
|   22   |   2010   | 120  |     60     |                                   Bumpy road                                    |
|   23   |   2010   | 150  |     60     |                                  Slippery road                                  |
|   24   |   2010   |  90  |     30     |                            Road narrows on the right                            |
|   25   |   2010   | 480  |    150     |                                    Road work                                    |
|   26   |   2010   | 180  |     60     |                                 Traffic signals                                 |
|   27   |   2010   |  60  |     30     |                                   Pedestrians                                   |
|   28   |   2010   | 150  |     60     |                                Children crossing                                |
|   29   |   2010   |  90  |     30     |                                Bicycles crossing                                |
|   30   |   2010   | 150  |     60     |                               Beware of ice/snow                                |
|   31   |   2010   | 270  |     90     |                              Wild animals crossing                              |
|   32   |   2010   |  60  |     30     |                       End of all speed and passing limits                       |
|   33   |   2010   | 210  |     90     |                                Turn right ahead                                 |
|   34   |   2010   | 120  |     60     |                                 Turn left ahead                                 |
|   35   |   2010   | 390  |    120     |                                   Ahead only                                    |
|   36   |   2010   | 120  |     60     |                              Go straight or right                               |
|   37   |   2010   |  60  |     30     |                               Go straight or left                               |
|   38   |   2010   | 690  |    210     |                                   Keep right                                    |
|   39   |   2010   |  90  |     30     |                                    Keep left                                    |
|   40   |   2010   |  90  |     60     |                              Roundabout mandatory                               |
|   41   |   2010   |  60  |     30     |                                End of no passing                                |
|   42   |   2010   |  90  |     30     |               End of no passing by vehicles over 3.5 metric tons                |

![alt text][image7]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer    |     Description	     |  stride    | Kernel     |Padding  |  output   |  
|:--------:|:---------------------:|:----------:|:----------:|:-------:|:---------:|
|          | Input image 32x32x1   |            |            |         |           | 
| 1      	 |  	Convolution        |     1x1    |   5x5x6    |  VALID  |  28x28x6  |
| 		     |     RELU              |		  			|            |         |           |
|          |  Max pooling	         |    2x2     |   2x2      |  VALID  | 14x14x6   |
| 2        |  Convolution  	      |    1x1      |   5x5x16   | VALID   | 10x10x16  | 
|          |      RELU     		    |   					|            |         |           |
|          |  Max pooling	        |    2x2			|    2x2     | VALID   | 5x5x16    |
|	3        | 	Convolution         |		1x1  			|    5x5x400 | VALID   | 1x1x400   |
|					 | 	     RELU           |							|            |         |           |
| 4        | flatten(layer2,layer3|             |            |         |  800      |
| 5        | Dropout              |             |            |         |  800      |
| 6        | Fully connected      |             |            |         |  43       |


 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

  * EPOCHS = 200
  * BATCH_SIZE = 128
  * rate = 0.0009
  * and Adam optimizer 


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

##### Step 1: evaluate the models
* In this step I train all models with a small number of epoch to evaluate the model performance, then to choose the final model. Chosen parameters of traning is:
  + Epoch: 10
  + Training rate: 0.0009
* Training LeNet:
<pre><code>
Training...

EPOCH 1 ...Validation accuracy = 0.798...training time: 6 s
EPOCH 2 ...Validation accuracy = 0.864...training time: 6 s
EPOCH 3 ...Validation accuracy = 0.896...training time: 5 s
EPOCH 4 ...Validation accuracy = 0.904...training time: 5 s
EPOCH 5 ...Validation accuracy = 0.904...training time: 5 s
EPOCH 6 ...Validation accuracy = 0.920...training time: 5 s
EPOCH 7 ...Validation accuracy = 0.919...training time: 6 s
EPOCH 8 ...Validation accuracy = 0.921...training time: 6 s
EPOCH 9 ...Validation accuracy = 0.921...training time: 5 s
EPOCH 10 ...Validation accuracy = 0.941...training time: 6 s
Model saved
Training time:  67  seconds

Test Accuracy = 0.912
Train Accuracy = 0.995
Validation Accuracy = 0.941
</code></pre>
![alt text][image8]



* Training Modified_LeNet
<pre><code>
Training...

EPOCH 1 ...Validation accuracy = 0.881...training time: 6 s
EPOCH 2 ...Validation accuracy = 0.919...training time: 6 s
EPOCH 3 ...Validation accuracy = 0.934...training time: 6 s
EPOCH 4 ...Validation accuracy = 0.942...training time: 6 s
EPOCH 5 ...Validation accuracy = 0.943...training time: 5 s
EPOCH 6 ...Validation accuracy = 0.949...training time: 5 s
EPOCH 7 ...Validation accuracy = 0.941...training time: 5 s
EPOCH 8 ...Validation accuracy = 0.951...training time: 6 s
EPOCH 9 ...Validation accuracy = 0.947...training time: 6 s
EPOCH 10 ...Validation accuracy = 0.944...training time: 7 s
Model saved
Training time:  79  seconds

Test Accuracy = 0.935
Train Accuracy = 0.999
Validation Accuracy = 0.944
</code></pre>
![alt text][image9]


* Training Modified_LeNet_1
<pre><code>
Training...

EPOCH 1 ...Validation accuracy = 0.849...training time: 7 s
EPOCH 2 ...Validation accuracy = 0.892...training time: 7 s
EPOCH 3 ...Validation accuracy = 0.905...training time: 7 s
EPOCH 4 ...Validation accuracy = 0.910...training time: 6 s
EPOCH 5 ...Validation accuracy = 0.924...training time: 7 s
EPOCH 6 ...Validation accuracy = 0.925...training time: 7 s
EPOCH 7 ...Validation accuracy = 0.930...training time: 7 s
EPOCH 8 ...Validation accuracy = 0.926...training time: 7 s
EPOCH 9 ...Validation accuracy = 0.924...training time: 6 s
EPOCH 10 ...Validation accuracy = 0.932...training time: 6 s
Model saved
Training time:  84  seconds

Test Accuracy = 0.916
Train Accuracy = 0.992
Validation Accuracy = 0.933
</code></pre>
![alt text][image10]

  
* Training Modified_LeNet_2
<pre><code>
Training...

EPOCH 1 ...Validation accuracy = 0.745...training time: 8 s
EPOCH 2 ...Validation accuracy = 0.826...training time: 8 s
EPOCH 3 ...Validation accuracy = 0.843...training time: 8 s
EPOCH 4 ...Validation accuracy = 0.855...training time: 8 s
EPOCH 5 ...Validation accuracy = 0.875...training time: 8 s
EPOCH 6 ...Validation accuracy = 0.877...training time: 7 s
EPOCH 7 ...Validation accuracy = 0.881...training time: 7 s
EPOCH 8 ...Validation accuracy = 0.885...training time: 8 s
EPOCH 9 ...Validation accuracy = 0.887...training time: 8 s
EPOCH 10 ...Validation accuracy = 0.887...training time: 8 s
Model saved
Training time:  100  seconds

Test Accuracy = 0.893
Train Accuracy = 0.983
Validation Accuracy = 0.886
</code></pre>
![alt text][image11]




My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

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

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


