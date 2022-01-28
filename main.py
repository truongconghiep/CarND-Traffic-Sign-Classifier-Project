from Traffic_Sign_Classifier import *
import glob
from lenet_traffic_sign_classifier import *

# data_pre_processing = data_augmentation(path_to_train_data = "./traffic-signs-data/train.p", \
#     path_to_validation_data = "./traffic-signs-data/valid.p", \
#         path_to_test_data = "./traffic-signs-data/test.p")



# data_pre_processing.augment_data()
# data_pre_processing.save_augmented_data("./traffic-signs-data/processed_train.p", \
#     "./traffic-signs-data/processed_valid.p", "./traffic-signs-data/processed_test.p")


# Load pre-processed image data
X_train, y_train = Read_Data_From_Pickle('./traffic-signs-data/PreprocessedTrainData.p')
X_valid, y_valid = Read_Data_From_Pickle('./traffic-signs-data/PreprocessedValidationData.p')
X_test, y_test = Read_Data_From_Pickle('./traffic-signs-data/PreprocessedTestData.p')

classifier = lenet_traffic_sign_classifier()

classifier.init_training_pipeline()

classifier.train(X_train, y_train, X_valid, y_valid)

classifier.evaluate_model(X_test, y_test, X_train, y_train, X_valid, y_valid)

# Number of images to load
number_Signs = 40

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
image_gry = Convert_Data_To_GrayScale(np.array(read_images))

# Read csv data      
SignName_SvcFileName = './signnames.csv'
sign_names = Read_Csv(SignName_SvcFileName, 1, ',')
classifier.test_model(image_gry, read_labels, number_Signs=number_Signs, sign_names=sign_names)

