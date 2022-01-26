'''
Created on 07.03.2018

@author: Hiep Truong
'''
import csv
import pickle
import datetime
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
import random
import math
import keras
# from scipy.misc import imread
from imageio import imread
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from beautifultable import BeautifulTable

K.set_image_data_format('channels_first')

###############################  Basic functions #########################################################################
def Read_Data_From_Pickle(file_name):
    """
        This function read data from a pickle
            Input:
                file_name: pickle file name
            Return:
                Sampes, lables
    """
    with open(file_name, mode='rb') as f:
        train = pickle.load(f)
    return train['features'], train['labels']

def Write_Data_To_Pickle(data, file_name):
    """
        This function writes a data to pickle
            Input:
                data: data to write
                file_name:
    """
    pickle.dump( data, open( file_name, "wb" ) )

def Read_Csv(file_name, column_idx , delimiter):
    """
        This function reads a column from a csv file
            input: 
                file_name: name of the csv file
                column_idx: the index of the column to be read
                delimiter: column delimiter symbol
            return: a list of data
    """
    data = []
    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for num,row in enumerate(reader):
            if num >= 1:
                data.append(row[column_idx])
    return data

def Calculate_Time_Diff_Up_To_Now_in_second(start_time):
    """
        This function calculate a time difference between a time point in the pass and now
            Input: 
                start_time: a time point in the pass
            return time difference in seconds
    """
    after  = datetime.datetime.now()
    return math.floor(((after - start_time).seconds))

def print_Table(header,data):
    """
        plot a table of data with a given header
            input param:
                header: 1D list
                data: 2D list
    """ 
    table_data = []
    number_rows = min(len(header),len(data))
    table = BeautifulTable(max_width=100)
    table.column_headers = header[0:number_rows:1]
    table_data = data[0:number_rows]  
    for n in range(len(table_data[0])):
        table.append_row([row[n] for row in table_data])
    print(table)

def Plot_Images(images, title = None):
    """
        This function plot an arbitrary number of images
            input: 
                images: a numpy array of images
                title: a title for the plot
    """
    image_number = len(images)
    fig, axs = plt.subplots(int(image_number / 5),5, figsize=(20, 4 * image_number/5))
    fig.suptitle(title, fontsize=18)
    axs = axs.ravel()    
    for n in range(image_number):
        axs[n].axis('off')
        if images[n].shape[2] == 1:
            axs[n].imshow(images[n].squeeze(), cmap='gray')
        else:
            axs[n].imshow(images[n])
    plt.show() 
    
def Plot_Curve(data, labels = None, title = None, xLabel = None, yLabel = None):
    """
        This function plots a curve diagram of datas
            Input: 
                data: a list of datas, where data[0] is on x-axis, the other elements are data on y-axis
                labels: a list of labels of datas
                title: tilte of the diagram
                xLable: label on x
                yLable: lable on y
    """
    plt.figure(figsize=(10,5))
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.title(title)
    for i in range(1,len(data)):
        plt.plot(data[0], data[i],label = labels[i - 1])
    plt.legend()
    plt.show()    
    
def Data_Visualisation(labels,class_name):
    """
        This function visualizes a given data set
            input:
                labels: data set to be visualized
                class_name: name of classes of the data set
    """
    number_samples = []
    table_Data = []
    for i in range(len(labels)):
        img = number_of_labels_per_class(labels[i])
        number_samples.append(img)
    header = ["labels", "Training", "Test", "Validation", "Class name" ]
    # visualize data in a table
    x = [i for i in range(len(class_name))]
    table_Data.append(x)
    table_Data = table_Data + number_samples
    table_Data.append(class_name)
    print_Table(header,table_Data)
    # Barchart
    width = 0.3
    plt.figure(figsize=(20,10))
    plt.ylabel('number of samples')
    plt.xlabel('labels')
    plt.title('data sets')
    x = np.array(x)
    p0=plt.bar(x - width, number_samples[0], width = width, color='g', label = "training")
    p1=plt.bar(x, number_samples[1], width = width, color='b', label = "test")
    p2=plt.bar(x + width, number_samples[2], width = width, color='r', label = "validation")
    plt.legend((p0[0], p1[0], p2[0]), ('train' , 'test', 'validation'))
    plt.show()
    
def Get_and_crop_n_image_randomly(images, number_Signs, labels):
    """
        This function extracts a number of images from a given image set
            input:
                images: a numpy array of images
                number_Signs: number of images to extract
                labels: labels of input image set
            return: a list of images
    """
    cropped_images = []
    chosen_labels = []
    # Get n signs randomly from the test set
    indices = random.sample(range(0,len(images)),number_Signs) 
    # Read and crop images
    for n in range(0,number_Signs):
        img = imread(images[indices[n]])
        resized_image = cv2.resize(img,(32,32))
        cropped_images.append(resized_image)
        chosen_labels.append(labels[indices[n]])
    return cropped_images, chosen_labels

def Get_n_images_randomly(images, number_images, labels, convert_to_array = False):
    chosen_images = []
    chosen_labels = []
    # Get n signs randomly from the test set
    indices = random.sample(range(0,len(images)),number_images)
    # Read and crop images
    for n in range(0,number_images):
        if convert_to_array == False:
            chosen_images.append(images[indices[n]])
        else: 
            chosen_images.append(img_to_array(images[indices[n]]))         
        chosen_labels.append(labels[indices[n]])
    return (chosen_images), (chosen_labels)
    
def number_of_class(labels):
    """
        This function returns the number of classes in a data set
    """
    return len(np.unique(labels))
            
def number_of_labels_per_class(labels):
    """
        This function counts number of samples per class of a data set
            input:
                labels: labels of the data set
            return: a list of number of samples of all classes
    """
    number_samples = []
    n_classes = number_of_class(labels)
    for n in range (n_classes):
        number_samples.append(np.count_nonzero(labels == n))
    return number_samples


def extract_data_Subset(X_train, Y_train, label):
    """
        This function extracts all samples of a class in a data set
            input: 
                X_train: training samples
                Y_train: training labels
                labels: label of the class, whose samples are being extracted
            return: a numpy array of all sample of the class
    """
    dataset_train = []
    for i in range(0, len(Y_train)):
        if Y_train[i] == label:
            img = img_to_array(X_train[i])
            dataset_train.append(img)
    return np.array(dataset_train)
    
def GenerateNewSubsetData(origin_Data, number_Generated_img, batch_size):
    """
        This function generates a new data subset from an input data
            input:
                origin_Data: input data as basic to generate new data
                number_Generated_img: number of images to generate
                batch_size: batch size
            return a set of modified images, which have the same content as the one of the 
                    original data set
    """
    generated_Images = []

    datagen = ImageDataGenerator(rotation_range=4, 
                                 width_shift_range=0.05, 
                                 height_shift_range=0.05, 
                                 zoom_range=0.05,
                                shear_range=0.05
                                )
    # fit parameters from data
    datagen.fit(origin_Data)

    X_batch = datagen.flow(origin_Data, batch_size=batch_size)
    for i, new_images in enumerate(X_batch):
        if i < number_Generated_img:
            new_image = array_to_img(new_images[0])
            new_image = np.array(new_image)
            generated_Images.append(new_image)
        else:
            break
    return generated_Images

def augment_brightness_1_image(image):
    """
        This function creates a new image by changing the brightness of the original image
            input:
                image: basic image to change brightness
            return: the modified image
    """
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def Image_Augmentation(Original_Dataset_Train, Original_Dataset_Labels):
    """
        This function augments a data set of images. 
        It generates new images by different methods, than add the new images to the original data set.
            input:
                Original_Dataset_Train: original data samples
                Original_Dataset_Labels: original data labels
            return: an augmented data set
    """
    result_Train = []
    result_labels = []
    # Brightness augmentation     
    brightness_augment = []
    for i in (Original_Dataset_Train):
        img = augment_brightness_1_image(i)
        brightness_augment.append(img)
    brightness_augment = np.array(brightness_augment)
    brightness_augment = Original_Dataset_Train
    # Other augmentation methods 
    # calculate number of classes
    n_class = number_of_class(Original_Dataset_Labels)
    # calculate number of samples per labels
    number_samples_per_class = number_of_labels_per_class(Original_Dataset_Labels)
    # calculate maximun number samples per a label
    max_sample_num = max(number_samples_per_class)
    #iterate over all labels
    for label in range(0,n_class):
        print("label ", label)
        generated_labels = []
        extracted_Train = extract_data_Subset(brightness_augment, Original_Dataset_Labels, label)
        number_of_new_images = int((max_sample_num - number_samples_per_class[label]))
        if number_of_new_images > 0:
            generated_Train = GenerateNewSubsetData(extracted_Train, number_of_new_images, number_samples_per_class[label])
            generated_labels = [label] * number_of_new_images
            result_Train = result_Train + generated_Train
            result_labels = result_labels + generated_labels
    # concatenate the generated data to the original data
    result_Train = np.concatenate((Original_Dataset_Train,np.array(result_Train)), axis=0)
    result_labels = np.concatenate((Original_Dataset_Labels,np.array(result_labels)), axis=0)
    return shuffle(result_Train, result_labels)

def Convert_Data_To_GrayScale(data):
    """
        This function converts an image data set in grayscale
            input: 
                data: input data set
            return: a numpy array of grayscale images
    """
    return np.sum(data/3, axis=3, keepdims=True)

def Data_Normalization(data):
    """
        This function normalises an image data set
            input:
                data: grayscale image data set
            return normalised images
    """
    return data/255 - 0.5


class data_augmentation:
    def __init__ (self, path_to_train_data, path_to_validation_data, path_to_test_data):
        self.X_train, self.y_train = Read_Data_From_Pickle(path_to_train_data)
        self.X_valid, self.y_valid = Read_Data_From_Pickle(path_to_validation_data)
        self.X_test, self.y_test = Read_Data_From_Pickle(path_to_test_data)
        # Read csv data      
        SignName_SvcFileName = './signnames.csv'
        self.sign_names = Read_Csv(SignName_SvcFileName, 1, ',')
        # visualize the original data set
        Data_Visualisation([self.y_train, self.y_test, self.y_valid], self.sign_names)

    def augment_data(self):
        Augmentation_start_time = datetime.datetime.now()
        # augment the data set
        X_train, y_train = Image_Augmentation(self.X_train, self.y_train)
        # visualize augmented data
        Data_Visualisation([self.y_train, self.y_test, self.y_valid], self.sign_names)

        print("converting to grayscale")
        # convert the data set to grayscale
        X_train = Convert_Data_To_GrayScale(X_train)
        X_test = Convert_Data_To_GrayScale(self.X_test)
        X_valid = Convert_Data_To_GrayScale(self.X_valid)

        # Normalize the grayscale data set
        print("normalizing")
        X_train = Data_Normalization(X_train)
        X_test = Data_Normalization(X_test)
        X_valid = Data_Normalization(X_valid)
        print("Augmentation time: ", Calculate_Time_Diff_Up_To_Now_in_second(Augmentation_start_time), " seconds")

    def save_augmented_data(self, train_filename, validation_filename, test_filename):
        Write_Data_To_Pickle({"features":self.X_train,"labels":self.y_train}, train_filename)
        Write_Data_To_Pickle({"features":self.X_valid,"labels":self.y_valid}, validation_filename)
        Write_Data_To_Pickle({"features":self.X_test,"labels":self.y_test}, test_filename)
    