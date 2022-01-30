from tensorflow.keras.models import load_model
from data_preprocessing import *
from tensorflow.keras.utils import to_categorical
import time
import glob


X_train, y_train = Read_Data_From_Pickle('./traffic-signs-data/train.p')
X_valid, y_valid = Read_Data_From_Pickle('./traffic-signs-data/valid.p')
X_test, y_test = Read_Data_From_Pickle('./traffic-signs-data/test.p')
model = load_model('AlexNet_Traffic_Sign_Classifier.h5')

model.summary()

X_test, y_test = Read_Data_From_Pickle('./traffic-signs-data/test.p')
y_test=to_categorical(y_test)

start_time = time.time()
model.evaluate(X_test, y_test)
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_time_milliSeconds = elapsed_time*1000
print("evaluation time in milliseconds is ",elapsed_time_milliSeconds)
# Number of images to load
number_Signs = 10

# Read file names from germman traffic sign test set
dirName ='./GTSRB/Final_Test/Images'
# Read image file names
images = glob.glob(dirName + '/*.ppm')
# Read the corresponding labels
SignName_SvcFileName = 'GT-final_test.csv'
sign_labels_in_test = Read_Csv(SignName_SvcFileName, 7, ';')
# get randomly images from the data set
read_images, read_labels, indices = Get_and_crop_n_image_randomly(images, number_Signs, sign_labels_in_test, dirName)
Plot_Images(read_images, 'Test images')
# Convert the read images to grayscale


# Read csv data      
SignName_SvcFileName = './signnames.csv'
sign_names = Read_Csv(SignName_SvcFileName, 1, ',')

read_images = np.array(read_images)
print("read_images shape: ", read_images.shape)

logit = model.predict(read_images)

print("max logit ", np.argmax(logit, 1))
print("read_labels ", read_labels)

