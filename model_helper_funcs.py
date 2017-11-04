
import random
from random import shuffle
import math
import gc #garbage collector


def randomize_data(data):
    """
    randomly re-arrange the data in the list 
    """
    # shuffle(data)
    l = len(data)
    for i, card in enumerate(data):
        swapi = random.randrange(i, l)
        data[i], data[swapi] = data[swapi], card

def convert_to_numeric(data):
    """
    converts the 40 length strings of the input data to numeric
    input is a list of tuples. Each tuple has a string of length 40 and label to which it belongs to.
    """
    #-- should labels also be changed numeric or are they already provided in numeric format ? 
    for element in data:
        insert_number(element[0]) 

def insert_number(arr):
    """
    replaces each alphabet with a number 
    """
    char_dict = {'A':1, 'B':2, 'C':3, 'D':4} # not used now 
    for index,char in enumerate(arr):
        #arr[index] = char_dict[char]  
        if char == 'A':
           arr[index] = 1
        elif char == 'B':
           arr[index] = 2
        elif char == 'C':
           arr[index] = 3
        elif char == 'D':
           arr[index] = 4
    return arr


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
    features = []
    labels = []
    for element in data:
        #print(element)
        features.append(element[0])
        labels.append(element[1][0])
    return features, labels   


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


"""
# for debugging

if __name__ == "__main__":
    s1 = 'AB'
    s2 = 'CD'
    s3 = 'AA'
    s4 = 'BD'
    l1 = list(s1)
    l2 = list(s2)
    l3 = list(s3)
    l4 = list(s4)

    a1 = (l1,1)
    a2 = (l2,2)
    a4 = (l2,3)
    a5 = (l2,4)

    a3 = []
    a3.append(a1)
    a3.append(a2)
    a3.append(a4)
    a3.append(a5)

    print ("--",a3)
    randomize_data(a3)
    print ("--",a3)

    convert_to_numeric(a3)
    print ("--",a3)

    x,y = seperate_data_lables(a3) 
    print ("***",x)
    print ("***",y)

    for batch_features, batch_labels in batches(1,x,y):
        #print (batch_features)
         print('!! : {},{}'.format(batch_features,batch_labels))
"""
