from Traffic_Sign_Classifier import *



data_pre_processing = data_augmentation(path_to_train_data = "./traffic-signs-data/train.p", \
    path_to_validation_data = "./traffic-signs-data/valid.p", \
        path_to_test_data = "./traffic-signs-data/test.p")



data_pre_processing.augment_data()
data_pre_processing.save_augmented_data("./traffic-signs-data/processed_train.p", \
    "./traffic-signs-data/processed_valid.p", "./traffic-signs-data/processed_test.p")


