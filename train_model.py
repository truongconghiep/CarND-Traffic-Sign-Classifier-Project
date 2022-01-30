from Traffic_Sign_Classifier import *
from AlexNet_Traffic_Sign_Classifier import *

X_train, y_train = Read_Data_From_Pickle('./traffic-signs-data/train.p')
X_valid, y_valid = Read_Data_From_Pickle('./traffic-signs-data/valid.p')
X_test, y_test = Read_Data_From_Pickle('./traffic-signs-data/test.p')

classifier = AlexNet_Traffic_Sign_Classifier()
classifier.training(X_train, y_train, X_valid, y_valid)