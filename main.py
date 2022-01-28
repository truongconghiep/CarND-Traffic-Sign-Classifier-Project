from Traffic_Sign_Classifier import *
import glob
from lenet_traffic_sign_classifier import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time

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
image_gry = Convert_Data_To_GrayScale(np.array(read_images))

# Read csv data      
SignName_SvcFileName = './signnames.csv'
sign_names = Read_Csv(SignName_SvcFileName, 1, ',')
classifier.test_model(image_gry, read_labels, number_Signs=number_Signs, sign_names=sign_names)


with tf.Session() as sess:
    saver = tf.train.import_meta_graph('lenet.meta')
    saver.restore(sess, "lenet")
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    prediction = graph.get_tensor_by_name("prediction:0")
    start_time = time.time()
    logit = sess.run(prediction, feed_dict={x: image_gry, keep_prob: 1.0})
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_milliSeconds = elapsed_time*1000
    print("code elapsed time in milliseconds is ",elapsed_time_milliSeconds)

    # visualise the read images and prediction results
    header = ["Sign number", "predicted label", "annotation"]
    sign_name = [sign_names[i] for i in logit]
    table_Data = []
    table_Data.extend([[i+1 for i in range(number_Signs)],logit,sign_name])
    print_Table(header,table_Data)

    accuracy = 0.0
    result = []
    for n in range(number_Signs):
        if logit[n] == int(read_labels[n]):
            accuracy = accuracy + (100./number_Signs)
            result.append("correct")
        else:
            result.append("wrong")

    print("Total accuracy = {} %".format(accuracy))
    # visualize the accuracy with a table
    header = ["expected label", "predicted label", "result"]
    table_Data = []
    table_Data.extend([read_labels,logit,result])
    print_Table(header,table_Data)

