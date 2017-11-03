import os

from label_classifier import get_label


def load(folder_path):
    all_strings = []
    for filename in os.listdir(folder_path):
        input_file = open(os.path.join(folder_path, filename))
        lines = input_file.readlines()
        for line in lines:
            line = line.rstrip()
            all_strings.append((list(line), get_label(line)))
    return all_strings


if __name__ == '__main__':
    for i in load("test_folder_01"):
        print i