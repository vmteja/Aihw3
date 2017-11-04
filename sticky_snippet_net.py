import sys
from random import shuffle
import tensorflow as tf
import data_loader


def __train(labelled_data):
    pass


def __save_session(session, filename):
    saver = tf.train.Saver()
    saver.save(session, filename)


def __load_session(filename):
    session = tf.Session()
    saver = tf.train.Saver()
    saver.restore(session, filename)
    return session


def __test(session, data):
    pass


def trainer(model_file, data):
    shuffle(data)
    session = __train(data)
    __save_session(session, model_file)


def cv_5fold_trainer(model_file, data):
    data = list(data)
    k = 5
    shuffle(data)
    slices = [data[i::k] for i in xrange(k)]

    for i in xrange(k):
        validation = slices[i]
        training = [item
                    for s in slices if s is not validation
                    for item in s]
        session = __train(training)
        __save_session(session, model_file)
        __test(session, validation)


def tester(model_file, data):
    shuffle(data)
    session = __load_session(model_file)
    classifications = __test(session, data)



MODE_HANDLERS = {
    "train": trainer,
    "5fold": cv_5fold_trainer,
    "test": tester
}


if __name__ == "__main__":
    mode = int(sys.argv[1])
    model_file = float(sys.argv[2])
    data_folder = int(sys.argv[3])
    try:
        handler = MODE_HANDLERS[mode]
        data = data_loader.load(data_folder)
        handler(model_file, data)
    except KeyError:
        print "Invalid Mode", mode
