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
[image10]:  ./write_up/TrainingModified_Lenet_1_Epoch10.jpg "Training Modified_LeNet_1 Epoch = 10"
[image11]:  ./write_up/TrainingModiied_Lenet_2Epoch10.jpg "Training Modified_LeNet_2 Epoch = 10"
[image12]:  ./write_up/Final_Training.jpg "Final training"
[image13]:  ./write_up/TestImages.jpg "Test images"

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

Based on the performance evaluation above I choose the "Modified_LeNet" model, which has highest accuracy as my final model
##### Step 2: Train the final model
* In this step I train my final model with bigger number of epochs: 
  + EPOCH = 200
  + Learing rate = 0.0009

<code><pre>
Training...

EPOCH 1 ...Validation accuracy = 0.863...training time: 7 s
EPOCH 2 ...Validation accuracy = 0.907...training time: 7 s
EPOCH 3 ...Validation accuracy = 0.926...training time: 6 s
EPOCH 4 ...Validation accuracy = 0.949...training time: 7 s
EPOCH 5 ...Validation accuracy = 0.944...training time: 7 s
EPOCH 6 ...Validation accuracy = 0.954...training time: 7 s
EPOCH 7 ...Validation accuracy = 0.953...training time: 6 s
EPOCH 8 ...Validation accuracy = 0.959...training time: 5 s
EPOCH 9 ...Validation accuracy = 0.947...training time: 6 s
EPOCH 10 ...Validation accuracy = 0.954...training time: 7 s
EPOCH 11 ...Validation accuracy = 0.952...training time: 6 s
EPOCH 12 ...Validation accuracy = 0.949...training time: 6 s
EPOCH 13 ...Validation accuracy = 0.959...training time: 6 s
EPOCH 14 ...Validation accuracy = 0.941...training time: 6 s
EPOCH 15 ...Validation accuracy = 0.956...training time: 6 s
EPOCH 16 ...Validation accuracy = 0.961...training time: 6 s
EPOCH 17 ...Validation accuracy = 0.963...training time: 5 s
EPOCH 18 ...Validation accuracy = 0.963...training time: 5 s
EPOCH 19 ...Validation accuracy = 0.971...training time: 7 s
EPOCH 20 ...Validation accuracy = 0.963...training time: 7 s
EPOCH 21 ...Validation accuracy = 0.967...training time: 7 s
EPOCH 22 ...Validation accuracy = 0.973...training time: 7 s
EPOCH 23 ...Validation accuracy = 0.961...training time: 6 s
EPOCH 24 ...Validation accuracy = 0.963...training time: 7 s
EPOCH 25 ...Validation accuracy = 0.971...training time: 6 s
EPOCH 26 ...Validation accuracy = 0.964...training time: 6 s
EPOCH 27 ...Validation accuracy = 0.967...training time: 6 s
EPOCH 28 ...Validation accuracy = 0.967...training time: 6 s
EPOCH 29 ...Validation accuracy = 0.968...training time: 6 s
EPOCH 30 ...Validation accuracy = 0.967...training time: 6 s
EPOCH 31 ...Validation accuracy = 0.965...training time: 6 s
EPOCH 32 ...Validation accuracy = 0.962...training time: 6 s
EPOCH 33 ...Validation accuracy = 0.970...training time: 6 s
EPOCH 34 ...Validation accuracy = 0.967...training time: 5 s
EPOCH 35 ...Validation accuracy = 0.974...training time: 5 s
EPOCH 36 ...Validation accuracy = 0.964...training time: 6 s
EPOCH 37 ...Validation accuracy = 0.955...training time: 6 s
EPOCH 38 ...Validation accuracy = 0.970...training time: 6 s
EPOCH 39 ...Validation accuracy = 0.974...training time: 7 s
EPOCH 40 ...Validation accuracy = 0.974...training time: 6 s
EPOCH 41 ...Validation accuracy = 0.970...training time: 6 s
EPOCH 42 ...Validation accuracy = 0.968...training time: 7 s
EPOCH 43 ...Validation accuracy = 0.971...training time: 6 s
EPOCH 44 ...Validation accuracy = 0.973...training time: 6 s
EPOCH 45 ...Validation accuracy = 0.970...training time: 6 s
EPOCH 46 ...Validation accuracy = 0.967...training time: 6 s
EPOCH 47 ...Validation accuracy = 0.973...training time: 6 s
EPOCH 48 ...Validation accuracy = 0.973...training time: 6 s
EPOCH 49 ...Validation accuracy = 0.968...training time: 5 s
EPOCH 50 ...Validation accuracy = 0.971...training time: 5 s
EPOCH 51 ...Validation accuracy = 0.967...training time: 5 s
EPOCH 52 ...Validation accuracy = 0.966...training time: 5 s
EPOCH 53 ...Validation accuracy = 0.968...training time: 5 s
EPOCH 54 ...Validation accuracy = 0.973...training time: 5 s
EPOCH 55 ...Validation accuracy = 0.968...training time: 5 s
EPOCH 56 ...Validation accuracy = 0.968...training time: 5 s
EPOCH 57 ...Validation accuracy = 0.974...training time: 5 s
EPOCH 58 ...Validation accuracy = 0.968...training time: 5 s
EPOCH 59 ...Validation accuracy = 0.970...training time: 5 s
EPOCH 60 ...Validation accuracy = 0.973...training time: 5 s
EPOCH 61 ...Validation accuracy = 0.968...training time: 5 s
EPOCH 62 ...Validation accuracy = 0.974...training time: 5 s
EPOCH 63 ...Validation accuracy = 0.966...training time: 5 s
EPOCH 64 ...Validation accuracy = 0.963...training time: 5 s
EPOCH 65 ...Validation accuracy = 0.974...training time: 5 s
EPOCH 66 ...Validation accuracy = 0.975...training time: 5 s
EPOCH 67 ...Validation accuracy = 0.976...training time: 5 s
EPOCH 68 ...Validation accuracy = 0.971...training time: 5 s
EPOCH 69 ...Validation accuracy = 0.971...training time: 5 s
EPOCH 70 ...Validation accuracy = 0.969...training time: 5 s
EPOCH 71 ...Validation accuracy = 0.969...training time: 5 s
EPOCH 72 ...Validation accuracy = 0.973...training time: 5 s
EPOCH 73 ...Validation accuracy = 0.976...training time: 5 s
EPOCH 74 ...Validation accuracy = 0.968...training time: 5 s
EPOCH 75 ...Validation accuracy = 0.971...training time: 5 s
EPOCH 76 ...Validation accuracy = 0.978...training time: 5 s
EPOCH 77 ...Validation accuracy = 0.968...training time: 5 s
EPOCH 78 ...Validation accuracy = 0.972...training time: 5 s
EPOCH 79 ...Validation accuracy = 0.969...training time: 5 s
EPOCH 80 ...Validation accuracy = 0.971...training time: 5 s
EPOCH 81 ...Validation accuracy = 0.970...training time: 5 s
EPOCH 82 ...Validation accuracy = 0.967...training time: 5 s
EPOCH 83 ...Validation accuracy = 0.970...training time: 5 s
EPOCH 84 ...Validation accuracy = 0.979...training time: 5 s
EPOCH 85 ...Validation accuracy = 0.978...training time: 5 s
EPOCH 86 ...Validation accuracy = 0.955...training time: 5 s
EPOCH 87 ...Validation accuracy = 0.968...training time: 5 s
EPOCH 88 ...Validation accuracy = 0.976...training time: 5 s
EPOCH 89 ...Validation accuracy = 0.975...training time: 5 s
EPOCH 90 ...Validation accuracy = 0.977...training time: 6 s
EPOCH 91 ...Validation accuracy = 0.976...training time: 5 s
EPOCH 92 ...Validation accuracy = 0.976...training time: 5 s
EPOCH 93 ...Validation accuracy = 0.969...training time: 5 s
EPOCH 94 ...Validation accuracy = 0.967...training time: 5 s
EPOCH 95 ...Validation accuracy = 0.960...training time: 5 s
EPOCH 96 ...Validation accuracy = 0.963...training time: 6 s
EPOCH 97 ...Validation accuracy = 0.976...training time: 6 s
EPOCH 98 ...Validation accuracy = 0.973...training time: 5 s
EPOCH 99 ...Validation accuracy = 0.972...training time: 6 s
EPOCH 100 ...Validation accuracy = 0.976...training time: 6 s
EPOCH 101 ...Validation accuracy = 0.964...training time: 5 s
EPOCH 102 ...Validation accuracy = 0.969...training time: 5 s
EPOCH 103 ...Validation accuracy = 0.975...training time: 5 s
EPOCH 104 ...Validation accuracy = 0.973...training time: 5 s
EPOCH 105 ...Validation accuracy = 0.974...training time: 5 s
EPOCH 106 ...Validation accuracy = 0.965...training time: 6 s
EPOCH 107 ...Validation accuracy = 0.972...training time: 6 s
EPOCH 108 ...Validation accuracy = 0.978...training time: 6 s
EPOCH 109 ...Validation accuracy = 0.973...training time: 6 s
EPOCH 110 ...Validation accuracy = 0.971...training time: 6 s
EPOCH 111 ...Validation accuracy = 0.972...training time: 6 s
EPOCH 112 ...Validation accuracy = 0.975...training time: 6 s
EPOCH 113 ...Validation accuracy = 0.971...training time: 6 s
EPOCH 114 ...Validation accuracy = 0.979...training time: 6 s
EPOCH 115 ...Validation accuracy = 0.973...training time: 6 s
EPOCH 116 ...Validation accuracy = 0.977...training time: 6 s
EPOCH 117 ...Validation accuracy = 0.967...training time: 6 s
EPOCH 118 ...Validation accuracy = 0.981...training time: 6 s
EPOCH 119 ...Validation accuracy = 0.974...training time: 6 s
EPOCH 120 ...Validation accuracy = 0.978...training time: 6 s
EPOCH 121 ...Validation accuracy = 0.969...training time: 5 s
EPOCH 122 ...Validation accuracy = 0.966...training time: 6 s
EPOCH 123 ...Validation accuracy = 0.970...training time: 6 s
EPOCH 124 ...Validation accuracy = 0.978...training time: 6 s
EPOCH 125 ...Validation accuracy = 0.977...training time: 6 s
EPOCH 126 ...Validation accuracy = 0.978...training time: 5 s
EPOCH 127 ...Validation accuracy = 0.973...training time: 6 s
EPOCH 128 ...Validation accuracy = 0.967...training time: 5 s
EPOCH 129 ...Validation accuracy = 0.978...training time: 5 s
EPOCH 130 ...Validation accuracy = 0.981...training time: 5 s
EPOCH 131 ...Validation accuracy = 0.970...training time: 6 s
EPOCH 132 ...Validation accuracy = 0.975...training time: 6 s
EPOCH 133 ...Validation accuracy = 0.973...training time: 6 s
EPOCH 134 ...Validation accuracy = 0.975...training time: 6 s
EPOCH 135 ...Validation accuracy = 0.970...training time: 6 s
EPOCH 136 ...Validation accuracy = 0.977...training time: 5 s
EPOCH 137 ...Validation accuracy = 0.970...training time: 6 s
EPOCH 138 ...Validation accuracy = 0.975...training time: 6 s
EPOCH 139 ...Validation accuracy = 0.979...training time: 6 s
EPOCH 140 ...Validation accuracy = 0.977...training time: 6 s
EPOCH 141 ...Validation accuracy = 0.976...training time: 6 s
EPOCH 142 ...Validation accuracy = 0.973...training time: 6 s
EPOCH 143 ...Validation accuracy = 0.969...training time: 6 s
EPOCH 144 ...Validation accuracy = 0.979...training time: 5 s
EPOCH 145 ...Validation accuracy = 0.980...training time: 5 s
EPOCH 146 ...Validation accuracy = 0.978...training time: 6 s
EPOCH 147 ...Validation accuracy = 0.974...training time: 6 s
EPOCH 148 ...Validation accuracy = 0.978...training time: 5 s
EPOCH 149 ...Validation accuracy = 0.976...training time: 5 s
EPOCH 150 ...Validation accuracy = 0.979...training time: 5 s
EPOCH 151 ...Validation accuracy = 0.974...training time: 5 s
EPOCH 152 ...Validation accuracy = 0.976...training time: 6 s
EPOCH 153 ...Validation accuracy = 0.978...training time: 5 s
EPOCH 154 ...Validation accuracy = 0.979...training time: 6 s
EPOCH 155 ...Validation accuracy = 0.976...training time: 5 s
EPOCH 156 ...Validation accuracy = 0.980...training time: 5 s
EPOCH 157 ...Validation accuracy = 0.979...training time: 5 s
EPOCH 158 ...Validation accuracy = 0.979...training time: 5 s
EPOCH 159 ...Validation accuracy = 0.978...training time: 6 s
EPOCH 160 ...Validation accuracy = 0.973...training time: 6 s
EPOCH 161 ...Validation accuracy = 0.979...training time: 6 s
EPOCH 162 ...Validation accuracy = 0.975...training time: 5 s
EPOCH 163 ...Validation accuracy = 0.974...training time: 5 s
EPOCH 164 ...Validation accuracy = 0.977...training time: 5 s
EPOCH 165 ...Validation accuracy = 0.978...training time: 6 s
EPOCH 166 ...Validation accuracy = 0.977...training time: 5 s
EPOCH 167 ...Validation accuracy = 0.973...training time: 6 s
EPOCH 168 ...Validation accuracy = 0.977...training time: 6 s
EPOCH 169 ...Validation accuracy = 0.976...training time: 6 s
EPOCH 170 ...Validation accuracy = 0.982...training time: 6 s
EPOCH 171 ...Validation accuracy = 0.971...training time: 6 s
EPOCH 172 ...Validation accuracy = 0.978...training time: 6 s
EPOCH 173 ...Validation accuracy = 0.978...training time: 6 s
EPOCH 174 ...Validation accuracy = 0.974...training time: 5 s
EPOCH 175 ...Validation accuracy = 0.974...training time: 6 s
EPOCH 176 ...Validation accuracy = 0.972...training time: 6 s
EPOCH 177 ...Validation accuracy = 0.976...training time: 6 s
EPOCH 178 ...Validation accuracy = 0.979...training time: 6 s
EPOCH 179 ...Validation accuracy = 0.974...training time: 5 s
EPOCH 180 ...Validation accuracy = 0.968...training time: 5 s
EPOCH 181 ...Validation accuracy = 0.978...training time: 6 s
EPOCH 182 ...Validation accuracy = 0.978...training time: 5 s
EPOCH 183 ...Validation accuracy = 0.981...training time: 5 s
EPOCH 184 ...Validation accuracy = 0.980...training time: 5 s
EPOCH 185 ...Validation accuracy = 0.982...training time: 6 s
EPOCH 186 ...Validation accuracy = 0.982...training time: 5 s
EPOCH 187 ...Validation accuracy = 0.979...training time: 5 s
EPOCH 188 ...Validation accuracy = 0.969...training time: 6 s
EPOCH 189 ...Validation accuracy = 0.970...training time: 5 s
EPOCH 190 ...Validation accuracy = 0.979...training time: 5 s
EPOCH 191 ...Validation accuracy = 0.968...training time: 5 s
EPOCH 192 ...Validation accuracy = 0.974...training time: 6 s
EPOCH 193 ...Validation accuracy = 0.977...training time: 5 s
EPOCH 194 ...Validation accuracy = 0.977...training time: 5 s
EPOCH 195 ...Validation accuracy = 0.979...training time: 6 s
EPOCH 196 ...Validation accuracy = 0.973...training time: 5 s
EPOCH 197 ...Validation accuracy = 0.974...training time: 6 s
EPOCH 198 ...Validation accuracy = 0.976...training time: 6 s
EPOCH 199 ...Validation accuracy = 0.978...training time: 6 s
EPOCH 200 ...Validation accuracy = 0.975...training time: 6 s
Model saved
Training time:  1228  seconds
</pre></code>
![alt text][image12]

My final model results were:
* training set accuracy of 1.000 (100 %)
* validation set accuracy of 0.975 (97.5 %)
* test set accuracy of 0.959 (95.9 %)

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web:

![alt text][image13]

As we see in the test images, the brightness changes through the images in a wide range, for example it is very light in the first image, but very dark in the sixth and tenth image. That is very difficut for the classifier to predict these images.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Sign number | predicted label |      annotation      |
|:-----------:|:---------------:|:--------------------:|
|      1      |       34        |   Turn left ahead    |
|      2      |        2        | Speed limit (50km/h) |
|      3      |       35        |      Ahead only      |
|      4      |       13        |        Yield         |
|      5      |        2        | Speed limit (50km/h) |
|      6      |        4        | Speed limit (70km/h) |
|      7      |       13        |        Yield         |
|      8      |       13        |        Yield         |
|      9      |        2        | Speed limit (50km/h) |
|     10      |       12        |    Priority road     |


The model was able to correctly guess 9 of the 10 traffic signs, which gives an accuracy of 90%. This compares favorably to the accuracy on the test set of 95,9 %

Total accuracy = 90.0 %

| expected label | predicted label | result  |
|:--------------:|:---------------:|:-------:|
|       34       |       34        | correct |
|       2        |        2        | correct |
|       35       |       35        | correct |
|       13       |       13        | correct |
|       2        |        2        | correct |
|       4        |        4        | correct |
|       13       |       13        | correct |
|       13       |       13        | correct |
|       2        |        2        | correct |
|       7        |       12        |  wrong  |

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


| sign num |                          softmax probabilities                           |  indices   |
|:--------:|:------------------------------------------------------------------------:|:----------:|
|    1     |                 1.00000 0.00000 0.00000 0.00000 0.00000                  | 34 0 1 2 3 |
|    2     |                 1.00000 0.00000 0.00000 0.00000 0.00000                  | 2 1 0 3 4  |
|    3     |                 1.00000 0.00000 0.00000 0.00000 0.00000                  | 35 0 1 2 3 |
|    4     |                 1.00000 0.00000 0.00000 0.00000 0.00000                  | 13 0 1 2 3 |
|    5     |                 1.00000 0.00000 0.00000 0.00000 0.00000                  | 2 0 1 3 4  |
|    6     |                 1.00000 0.00000 0.00000 0.00000 0.00000                  | 4 0 1 2 3  |
|    7     |                 1.00000 0.00000 0.00000 0.00000 0.00000                  | 13 0 1 2 3 |
|    8     |                 1.00000 0.00000 0.00000 0.00000 0.00000                  | 13 0 1 2 3 |
|    9     |                 1.00000 0.00000 0.00000 0.00000 0.00000                  | 2 0 1 3 4  |
|    10    |                 1.00000 0.00000 0.00000 0.00000 0.00000                  | 12 0 1 2 3 |


For all images the model is very confident about the prediction results, however the prediction of the tenth image is wrong. The reason of the wrong prediction might be the low brightness on the image

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


