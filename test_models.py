from tensorflow.keras.models import load_model
from data_preprocessing import *
from tensorflow.keras.utils import to_categorical
import time
import glob
from lenet_traffic_sign_classifier import *


# Number of images to load
number_Signs = 12600

# Read file names from germman traffic sign test set
dirName ='./GTSRB/Final_Test/Images'
# Read image file names
images = glob.glob(dirName + '/*.ppm')
print("ytest size ", len(images))
# Read the corresponding labels
SignName_SvcFileName = 'GT-final_test.csv'
sign_labels_in_test = Read_Csv(SignName_SvcFileName, 7, ';')
# get randomly images from the data set
read_images, read_labels, indices = Get_and_crop_n_image_randomly(images, number_Signs, sign_labels_in_test, dirName)
# Plot_Images(read_images, 'Test images')
# Convert the read images to grayscale
image_gry = Convert_Data_To_GrayScale(np.array(read_images))


# Read csv data      
SignName_SvcFileName = './signnames.csv'
sign_names = Read_Csv(SignName_SvcFileName, 1, ',')

# Test AlexNet
model = load_model('AlexNet_Traffic_Sign_Classifier.h5')
start_time = time.time()
logits = model.predict(read_images)
logits = np.argmax(logits, 1)
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_time_milliSeconds = elapsed_time*1000
print("evaluation time in milliseconds is ",elapsed_time_milliSeconds)
correct = 0
for i in range(number_Signs):
    if logits[i] == int(read_labels[i]):
        correct = correct + 1
print("correct ", correct/number_Signs * 100)


# Test LENET
classifier = lenet_traffic_sign_classifier()
classifier.load_pretrained_model()
start_time = time.time()
classifier.test_model(image_gry, read_labels, number_Signs, sign_names, using_pretrainend=True, enable_table_visualization=False)
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_time_milliSeconds = elapsed_time*1000
print("evaluation time in milliseconds is ",elapsed_time_milliSeconds)

