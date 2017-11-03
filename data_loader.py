import os

from label_classifier import get_label


def load(folder_path):
    all_strings = []
    for filename in os.listdir(folder_path):
        input_file = open(filename)
        lines = input_file.readlines()
        for line in lines:
            all_strings.append((line, get_label(line)))
    return all_strings