
import tensorflow as tf
import math
from model_helper_funcs import * 
from data_loader import *

batch_size = 64 
no_epochs = 10

def dense_nn(X):
    """
    creates a 5 layer dense neural network, all hidden layers with ReLU activation
    while last layer has softmax activation 
    """
    layer_1_out = tf.contrib.layers.fully_connected(X,40, activation_fn=tf.nn.relu)
    layer_2_out = tf.contrib.layers.fully_connected(layer_1_out, 40, activation_fn = tf.nn.relu)
    layer_3_out = tf.contrib.layers.fully_connected(layer_2_out, 20, activation_fn = tf.nn.relu)
    layer_4_out = tf.contrib.layers.fully_connected(layer_3_out, 10, activation_fn = tf.nn.relu)
    layer_5_out = tf.contrib.layers.fully_connected(layer_4_out, 6,  activation_fn = None)
    return layer_5_out 


# Features and Labels
n_input = 40  # input array length
n_classes = 6 # no of classes 
# the 'None' in dimension below takes care of  the size of the batch 
features = tf.placeholder(tf.float32, [None, n_input]) 
labels = tf.placeholder(tf.float32, [None, n_classes])

# network architecuture
with tf.variable_scope('nn'):
     y_pred = dense_nn(features)
    
# training parameters
learning_rate = 0.01

# Define loss and optimizer
with tf.variable_scope('loss'):
     cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=labels))
     optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def train_model(model_filename, data):

    # randomly rearranging the input list 
    # randomize_data(data)

    # converting to numeric 
    convert_to_numeric(data)

    # split data into train and valid sets 
    train_data, valid_data = create_train_valid(data, 0.9)

    # size of train data 
    data_size = len(train_data)

    # deleting the 'data' variable and calling garbage collector to clear up memory
    del data
    gc.collect() 

    no_batches = math.ceil(data_size/batch_size)

    # seperating data elements and their labels 
    train_features, train_labels = seperate_data_lables(train_data)
    valid_features, valid_labels = seperate_data_lables(valid_data)

    # deleting the 'data' variable and calling garbage collector to clear up memory
    del train_data
    del valid_data
    gc.collect() 

    save_file = model_filename
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
         sess.run(init)

         # Training cycle
         for epoch in range(no_epochs):
             batch_count = 0 
             # Loop over all batches
             for batch_features, batch_labels in batches(batch_size, train_features, train_labels):
                 sess.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})
                 batch_count += 1
                 # printing status for every 5 batches 
                 if batch_count%20 == 0:
                    percent_done = (batch_count/no_batches)*100
                    #print "trained model with %d percent  of total batches" %(percent_done) # replace it with progress bar 
                
             # printing model's performance for every epoch 
             # calculate accuracy for validation dataset
             valid_accuracy = sess.run(accuracy, feed_dict={features: valid_features, labels: valid_labels})
             print('Epoch {:<1} - Validation Accuracy: {}'.format(epoch, valid_accuracy))
         print("--- trainig complete ----")

         # Save the model
         with tf.variable_scope('', reuse=True):
              saver.save(sess, save_file)
              print('Trained Model Saved.')
    return sess


def test_model(saved_file, data):

    # converting to numeric 
    convert_to_numeric(data)

    # size of train data 
    data_size = len(data)

    # seperating data elements and their labels 
    test_features, test_labels = seperate_data_lables(data)

    with tf.Session() as session:
        with tf.variable_scope('', reuse=True):
            #session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(session, saved_file)
            test_accuracy = session.run(accuracy, feed_dict={features: test_features, labels:test_labels })
            print('Test Accuracy: {}'.format(test_accuracy))
            #print session.run(accuracy, {X: INPUT_DATA_SET, Y: OUTPUT_DATA_SET})



def train(train_features, train_labels, valid_features, valid_labels):
    return time_taken, validation_accuracy

# for testing 
if __name__ == "__main__":
   
   #dir_path = "/home/rocky/cs256_assign/Aihw3/train"
   dir_path = "train"
   data = load(dir_path)
   #print (data[:5])
    
   train_model(data)

   file_name = 'train_model.ckpt.meta'
   test_model(file_name, data)

