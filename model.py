
import tensorflow as tf
import math

batch_size = 64 
no_epochs = 100

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

def batches(batch_size, features, labels):
    """
    creates batches of features and labels
    returns: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    output_batches = []

    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        output_batches.append(batch)
        
    return output_batches


def classifier(data):

    # randomly rearranging the input list 
    data = randomize_data(data)

    # converting to numeric 
    data = convert_to_numeric(data)

    # split data into train and valid sets 
    train_data, valid_data = create_train_valid(data, 0.9)

    # seperating data elements and their labels 
    train_features, train_labels = seperate_data_lables(train_data)
    valid_features, valid_labels = seperate_data_lables(valid_data)

    # Features and Labels
    n_input = 40  # input array lenght 
    n_classes = 6 # no of classes 
    # the 'None' in dimension below takes care of  the size of the batch 
    features = tf.placeholder(tf.float32, [None, n_input]) 
    labels = tf.placeholder(tf.float32, [None, n_classes])

    # network architecuture
    y_pred = dense_nn(features)

    # training parameters
    learning_rate = 0.01
    data_size = len(data)
    no_bacthes = math.ceil(data_size/batch_size)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=labels))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost) 

    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    save_file = './train_model.ckpt'
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
                 if batch_count%5 == 0:
                    percent_done = (batch_count/no_batches)*100
                    print ("trained model with {}% of batches",percent_done) # replace it with progress bar 
                
             # printing model's performance for every epoch 
             # calculate accuracy for validation dataset
             valid_accuracy = sess.run(accuracy, feed_dict={features: valid_features, labels: valid_labels})
             print('Epoch {:<3} - Validation Accuracy: {}'.format(epoch, valid_accuracy))
         print("--- trainig complete ----")

    # Save the model
    saver.save(sess, save_file)
    print('Trained Model Saved.') 
