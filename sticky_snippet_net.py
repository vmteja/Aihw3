import sys

import data_loader


def trainer(model_file, data):
    pass


def cv_5fold_trainer(model_file, data):
    pass


def tester(model_file, data):
    pass



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
