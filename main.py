from Traffic_Sign_Classifier import *
import glob
from lenet_traffic_sign_classifier import *

# data_pre_processing = data_augmentation(path_to_train_data = "./traffic-signs-data/train.p", \
#     path_to_validation_data = "./traffic-signs-data/valid.p", \
#         path_to_test_data = "./traffic-signs-data/test.p")



# data_pre_processing.augment_data()
# data_pre_processing.save_augmented_data("./traffic-signs-data/processed_train.p", \
#     "./traffic-signs-data/processed_valid.p", "./traffic-signs-data/processed_test.p")

# Number of images to load
number_Signs = 10

# Read file names from germman traffic sign test set
dirName ='./GTSRB/Final_Test/Images'
# Read image file names
images = glob.glob(dirName + '/*.ppm')
# Read the corresponding labels
SignName_SvcFileName = './GTSRB/Final_Test/GT-final_test.csv'
sign_labels_in_test = Read_Csv(SignName_SvcFileName, 7, ';')
# get randomly images from the data set
read_images, read_labels = Get_and_crop_n_image_randomly(images, number_Signs, sign_labels_in_test)
Plot_Images(read_images, 'Test images')
# Convert the read images to grayscale
image_gry = Convert_Data_To_GrayScale(np.array(read_images))

traffic_sign_classifier = lenet_traffic_sign_classifier()
traffic_sign_classifier.init_training_pipeline()
traffic_sign_classifier.test_model_and_run_test('lenet.meta', read_images, read_labels, sign_names= sign_labels_in_test)