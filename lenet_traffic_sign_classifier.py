from unicodedata import name
from Traffic_Sign_Classifier import *
from tensorflow.keras.layers import Flatten
import glob
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class lenet_traffic_sign_classifier:
    def __init__ (self, mu=0, sigma=0.1, num_of_channel = 1, num_of_output_class = 43, \
                                        epochs = 200, batch_size = 128, rate = 0.0009):
        self.mu = 0
        self.sigma = 0.1
        self.num_of_channel = num_of_channel
        self.num_of_output_class = num_of_output_class
        self.epoch = epochs
        self.batch_size = batch_size
        self.rate = rate


    def convolution(self, x, shape = None, strides = None, padding = 'VALID'):
        conv_W = tf.Variable(tf.truncated_normal(shape=shape, mean = self.mu, stddev = self.sigma))
        conv_b = tf.Variable(tf.zeros(shape[3]))
        return tf.nn.conv2d(x, conv_W, strides=strides, padding=padding) + conv_b
                           
    def fully_Connected(self, x, input_size, output_size, mu = 0, sigma = 0.1):
        fc_W = tf.Variable(tf.truncated_normal(shape=(input_size, output_size), mean = mu, stddev = sigma))
        fc_b = tf.Variable(tf.zeros(output_size))
        return tf.matmul(x, fc_W)  + fc_b

    def LeNet(self, x, mu, sigma, number_channels, number_ouput_class):    
        # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        conv1 =  self.convolution(x=x, shape = (5, 5, number_channels, 6), strides = [1, 1, 1, 1])                      
        # Activation.
        conv1 = tf.nn.relu(conv1)
        # Pooling. Input = 28x28x6. Output = 14x14x6.
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # Layer 2: Convolutional. Output = 10x10x16.
        conv2 = self.convolution(conv1, shape = (5, 5, 6, 16), strides = [1, 1, 1, 1])
        # Activation.
        conv2 = tf.nn.relu(conv2)
        # Pooling. Input = 10x10x16. Output = 5x5x16.
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # Flatten. Input = 5x5x16. Output = 400.
        fc0   = Flatten(conv2)
        # Layer 3: Fully Connected. Input = 400. Output = 120.
        fc1   = self.fully_Connected(fc0,400,120)                        
        # Activation.
        fc1  = tf.nn.relu(fc1)
        # Layer 4: Fully Connected. Input = 120. Output = 84.
        fc2    = self.fully_Connected(fc1,120,84)
        # Activation.
        fc2    = tf.nn.relu(fc2)
        # Layer 5: Fully Connected. Input = 84. Output = 43
        logits = self.fully_Connected(fc2,84,number_ouput_class)

        return logits

    def Modified_LeNet(self, x, mu, sigma, number_channels, number_ouput_class):
        
        # Layer 1: Convolutional. Input = 32x32xnumber_channels. Output = 28x28x6.  
        x = self.convolution(x, shape=(5, 5, 1, 6), strides=[1, 1, 1, 1])
        # Activation.
        x = tf.nn.relu(x)    
        # Pooling. Input = 28x28x6. Output = 14x14x6.
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        layer1 = x    
        # Layer 2: Convolutional. Output = 10x10x16.
        x = self.convolution(x, shape = (5, 5, 6, 16), strides = [1, 1, 1, 1])                     
        # Activation.
        x = tf.nn.relu(x)
        # Pooling. Input = 10x10x16. Output = 5x5x16.
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')    
        # Flatten. Input = 5x5x16. Output = 400.
        layer2flat = Flatten()(x)    
        # Layer 3: Convolutional. Output = 1x1x400.   
        x = self.convolution(x, shape = (5, 5, 16, 400), strides = [1, 1, 1, 1])                     
        # Activation.
        x = tf.nn.relu(x)
        # Flatten x. Input = 1x1x400. Output = 400.
        layer3flat = Flatten()(x)    
        # Concat layer2flat and x. Input = 400 + 400. Output = 800
        x = tf.concat([layer3flat, layer2flat], 1)    
        # Dropout
        x = tf.nn.dropout(x, self.keep_prob)    
        # Layer 4: Fully Connected. Input = 800. Output = number_ouput_class.
        logits = self.fully_Connected(x,800,number_ouput_class)
        
        return logits

    def Modified_LeNet1(self, x, mu, sigma, number_channels, number_ouput_class):
        
        # Layer 1: Convolutional. Input = 32x32xnumber_channels. Output = 28x28x6.  
        x = self.convolution(x, shape = (5, 5, number_channels, 6), strides = [1, 1, 1, 1])
        # Activation.
        x = tf.nn.relu(x)
        # Pooling. Input = 28x28x6. Output = 14x14x6.
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        Stage1 = x
        ############## Stage 2 ##########################
        # Layer 2: Convolutional. Output = 10x10x16.
        x = self.convolution(x, shape = (5, 5, 6, 16), strides = [1, 1, 1, 1])               
        # Activation.
        x = tf.nn.relu(x)
        # Pooling. Input = 10x10x16. Output = 5x5x16.
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        x_branched = self.convolution(Stage1, shape = (5, 5, 6, 16), strides = [1, 2, 2, 1])
        ########################### stage 3 ###########
        # Flatten. Input = 5x5x16. Output = 400.
    #     layer2flat = flatten(x)
        # Layer 3: Convolutional. Output = 1x1x400.   
        x = self.convolution(x, shape = (5, 5, 16, 400), strides = [1, 1, 1, 1])
        x_branched = self.convolution(x_branched, shape = (5, 5, 16, 400), strides = [1, 1, 1, 1])              
        # Activation.
        x = tf.nn.relu(x)
        # Flatten x. Input = 1x1x400. Output = 400.
        layer3flat = Flatten(x)
        layer3flat_branched = Flatten(x_branched)
        # Concat layer2flat and x. Input = 400 + 400. Output = 800
        x = tf.concat([layer3flat, layer3flat_branched], 1)
        # Dropout
        x = tf.nn.dropout(x, 0.5)
        # Layer 4: Fully Connected. Input = 800. Output = number_ouput_class.
        logits = self.fully_Connected(x,800,number_ouput_class)
        
        return logits

    def Modified_LeNet2(self, x, mu, sigma, number_channels, number_ouput_class):    
        # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x12.
        conv1 =  self.convolution(x, shape = (5, 5, number_channels, 12), strides = [1, 1, 1, 1])                      
        # Activation.
        conv1 = tf.nn.relu(conv1)
        # Pooling. Input = 28x28x12. Output = 14x14x12.
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # Dropout
        conv1 = tf.nn.dropout(conv1, 0.7)
        conv1a = conv1
        conv1a_mp = tf.nn.max_pool(conv1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv1a_mp = Flatten(conv1a_mp)
        # Layer 2: Convolutional. Output = 10x10x24.
        conv2 = self.convolution(conv1, shape = (5, 5, 12, 24), strides = [1, 1, 1, 1])
        # Activation.
        conv2 = tf.nn.relu(conv2)
        # Pooling. Input = 10x10x16. Output = 5x5x24.
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # Dropout
        conv2 = tf.nn.dropout(conv2, 0.6)
        # Flatten. Input = 5x5x16. Output = 400.
        fc0   = Flatten(conv2)
        x = tf.concat([fc0, conv1a_mp], 1)                
        # fully connected output = 320
        fc1 = self.fully_Connected(x,1188,320)
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, 0.6)
        logits = self.fully_Connected(fc1,320,number_ouput_class)

        return logits

    def evaluate(self, X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, self.batch_size):
            batch_x, batch_y = X_data[offset:offset+self.batch_size], y_data[offset:offset+self.batch_size]
            accuracy = sess.run(self.accuracy_operation, feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: 1.0})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples
    
    def init_training_pipeline(self):
        self.x = tf.placeholder(tf.float32, (None, 32, 32, self.num_of_channel), name="x")
        self.y = tf.placeholder(tf.int32, (None), name="y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob") # probability to keep units
        one_hot_y = tf.one_hot(self.y, self.num_of_output_class)

        logits = self.Modified_LeNet(self.x, self.mu, self.sigma, self.num_of_channel, self.num_of_output_class)
        self.prediction = tf.argmax(logits, 1, name = "prediction")
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
        loss_operation = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate = self.rate)
        self.training_operation = optimizer.minimize(loss_operation)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.saver = tf.train.Saver()

    def train(self, X_train, y_train, X_valid, y_valid):

        before = datetime.datetime.now()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = len(X_train)
            
            validation_accuracy_figure = []
            
            print("Training...")
            print()
            for i in range(self.epoch):
                epoch_start_time = datetime.datetime.now()
                X_train, y_train = shuffle(X_train, y_train)
                for offset in range(0, num_examples, self.batch_size):
                    end = offset + self.batch_size
                    batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                    sess.run(self.training_operation, feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: 0.5})
                    
                validation_accuracy = self.evaluate(X_valid, y_valid)
                validation_accuracy_figure.append(validation_accuracy)
                epoch_training_time = Calculate_Time_Diff_Up_To_Now_in_second(epoch_start_time)
                print("EPOCH {} ...Validation accuracy = {:.3f}...training time: {} s"
                    .format(i+1, validation_accuracy, epoch_training_time))
                
            self.saver.save(sess, './lenet')
            print("Model saved")
        # Calculate the training time               
        print("Training time: ", Calculate_Time_Diff_Up_To_Now_in_second(before), " seconds")
        # Plot accuracies on a diagram
        plot_data = []
        plot_data.append([i for i in range (0, self.epoch)])
        plot_data.append(validation_accuracy_figure)
        labels = ['Training accuracy', 'Validation accuracy']
        Plot_Curve(plot_data, labels, title = "Accuracy", xLabel = "EPOCH", yLabel = "Accuracy")

    def evaluate_model(self, X_test, y_test, X_train, y_train, X_valid, y_valid):
        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('.'))

            test_accuracy = self.evaluate(X_test, y_test)
            print("Test Accuracy = {:.3f}".format(test_accuracy))  
            Train_accuracy = self.evaluate(X_train, y_train)
            print("Train Accuracy = {:.3f}".format(Train_accuracy))
            Validation_accuracy = self.evaluate(X_valid, y_valid)
            print("Validation Accuracy = {:.3f}".format(Validation_accuracy))

    def test_model(self, X_test, y_test, number_Signs = 10, sign_names = []):
        logit = []
        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('.'))
            logit = sess.run(self.prediction, feed_dict={self.x: X_test, self.keep_prob: 1.0})

        # visualise the read images and prediction results
        header = ["Sign number", "predicted label", "annotation"]
        print(logit)
        sign_name = [sign_names[i] for i in logit]
        table_Data = []
        table_Data.extend([[i+1 for i in range(number_Signs)],logit,sign_name])
        print_Table(header,table_Data)

        accuracy = 0.0
        result = []
        for n in range(number_Signs):
            if logit[n] == int(y_test[n]):
                accuracy = accuracy + (100./number_Signs)
                result.append("correct")
            else:
                result.append("wrong")

        print("Total accuracy = {} %".format(accuracy))
        # visualize the accuracy with a table
        header = ["expected label", "predicted label", "result"]
        table_Data = []
        table_Data.extend([y_test,logit,result])
        print_Table(header,table_Data)


    
