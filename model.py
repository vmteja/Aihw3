
import tensorflow as tf
import math

mini_batch_size = 64 

def dense_nn(X):
    """
    creates a 5 layer dense neural network, all hidden layers with ReLU activation
    while last layer has softmax activation 
    """
    layer_1_out = tf.contrib.layers.fully_connected(X,40, activation_fn=tf.nn.relu)
    layer_2_out = tf.contrib.layers.fully_connected(layer_1_out,40, activation_fn=tf.nn.relu)
    layer_3_out = tf.contrib.layers.fully_connected(layer_2_out,20, activation_fn=tf.nn.relu)
    layer_4_out = tf.contrib.layers.fully_connected(layer_3_out,10, activation_fn=tf.nn.relu)
    layer_5_out = tf.contrib.layers.fully_connected(layer_4_out,6, activation_fn=tf.nn.softmax)
    return layer_5_out 

def randomize_data(data):
    """
    randomly re-arrange the data in the list 
    """
    pass 

def convert_to_numeric(data):
    """
    converts the 40 length strings of the input data to numeric
    input is a list of tuples. Each tuple has a string of length 40 and label to which it belongs to.
    returns  a list of tuples 
    """
    #-- should labels also be changed numeric or are they already provided in numeric format ? 
    pass 

def create_train_valid(data, split_fraction):
    """
    splits the data into train and validation sets 
    """
    l = len(data)
    split_size = int(l*split_fraction)
    train_data = data[:split_size]
    valid_data = data[split_size:]
    return train_data, valid_data

def seperate_data_lables(data):
    """
    seperates data and their respective labels 
    returns two lists; one a list of data elements (40-leng numeric arrays),
    second list is the labels at their their corresponding indexes 
    """
    pass


def classifier(data):

    '''
    todo:
     1. read the input data and convert them to integers 
     2. create batches 
     3. seperate data and labels
     4. create train data and validation data  
    '''
    # randomly rearranging the input list 
    data = randomize_data(data)

    # converting to numeric 
    data = convert_to_numeric(data)

    # split data into train and valid sets 
    train_data, valid_data = create_train_valid(data, 0.9)

    # seperating data elements and their labels 
    train_x, train_y = seperate_data_lables(train_data)
    valid_x, valid_y = seperate_data_lables(valid_data)

    # declare Model's input and output 
    # what should be the dimensions if batches are passed ? 
    X = tf.placeholder(tf.float32, shape=(2,1))
    y_pred = tf.placeholder(tf.float32, shape=(6))

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
