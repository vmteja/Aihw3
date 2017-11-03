

import tensorflow as tf
import math

mini_batch_size = 64 

# creates batches from  given dataset 
def create_batches():
    pass

# creates a 5 layer dense neural network 
def dense_nn(X):
    layer_1_out = tf.contrib.layers.fully_connected(inputs,40, activation_fn=tf.nn.relu)
    layer_2_out = tf.contrib.layers.fully_connected(layer_1_out,40, activation_fn=tf.nn.relu)
    layer_3_out = tf.contrib.layers.fully_connected(layer_2_out,20, activation_fn=tf.nn.relu)
    layer_4_out = tf.contrib.layers.fully_connected(layer_3_out,10, activation_fn=tf.nn.relu)
    layer_5_out = tf.contrib.layers.fully_connected(layer_4_out,6, activation_fn=tf.nn.softmax)
    return layer_5_out 

def session_optimizer():
    pass 


def dense_layer(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us


def classifier():

    '''
    todo:
     1. read the input data and convert them to integers 
     2. create batches 
     3. seperate data and labels
     4. create train data and validation data  
    #get the data and its labels 
    #X_data = 
    #Y_labels = 
    
    #split the data into training and validation sets
    split_size = int(X_data.shape[0]*0.8)
    train_x, val_x = X_data[:split_size], X_data[split_size:]
    train_y, val_y = Y_labels[:split_size], Y_labels[split_size:]
    ''' 
 
    # Model input and output 
    # what should be the dimensions if batches are passed ? 
    X = tf.placeholder(tf.float32, shape=(2,1))
    Y = tf.placeholder(tf.float32, shape=(6))

    '''
    # layer-1 parameters
    w1 = tf.get_variable("w1", shape=(2,2), initializer = uniform_init)
    b1 = tf.get_variable("b1", shape=(2,1), initializer = uniform_init)
    '''

    # network architecuture
    y_pred = dense_nn(X)

    # loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=y_pred)) # compute costs

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    save_file = './train_model.ckpt'
    batch_size = 128
    n_epochs = 100

    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
         sess.run(tf.global_variables_initializer())

         # Training cycle
         for epoch in range(n_epochs):

             # Loop over all batches
             for i in range(total_batch):
                 sess.run(
                          optimizer,
                          feed_dict={features: data_x, labels: data_y)

             # Print status for every epochs
             valid_accuracy = sess.run(
                                       accuracy,
                                       feed_dict={
                                                  features: val_x
                                                 labels: val_y})
            print('Epoch {:<3} - Validation Accuracy: {}'.format(
                epoch,
                valid_accuracy))

    # Save the model
    saver.save(sess, save_file)
    print('Trained Model Saved.') 
