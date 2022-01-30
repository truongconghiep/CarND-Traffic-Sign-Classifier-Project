from data_preprocessing import *
from AlexNet_Traffic_Sign_Classifier import *

X_train, y_train = Read_Data_From_Pickle('./traffic-signs-data/rgb_processed_train.p')
X_valid, y_valid = Read_Data_From_Pickle('./traffic-signs-data/rgb_processed_valid.p')
X_test, y_test = Read_Data_From_Pickle('./traffic-signs-data/rgb_processed_test.p')


classifier = AlexNet_Traffic_Sign_Classifier(epochs=200)
classifier.training(X_train, y_train, X_valid, y_valid)