import tensorflow as tf
import numpy as np
import pathlib
import datetime

from Traffic_Sign_Classifier import *

# printout versions
print(f"Tensor Flow Version: {tf.__version__}")
print(f"numpy Version: {np.version.version}")

data_pre_processing = data_augmentation(path_to_train_data = "./traffic-signs-data/train.p", \
    path_to_validation_data = "./traffic-signs-data/valid.p", \
        path_to_test_data = "./traffic-signs-data/test.p")

data_pre_processing.augment_data(gray_scale=False)
data_pre_processing.save_augmented_data("./traffic-signs-data/rgb_processed_train.p", \
    "./traffic-signs-data/rgb_processed_valid.p", "./traffic-signs-data/rgb_processed_test.p")