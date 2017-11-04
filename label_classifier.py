

LABELS = [
    "NONSTICK", "12-STICKY", "34-STICKY", "56-STICKY", "78-STICKY", "STICK_PALINDROME"
]


class LabelConverter():
    def __init__(self):
        pass

    @classmethod
    def integer_to_onehot(cls, integer_label):
        id_list = [0] * len(LABELS)
        id_list[integer_label] = 1
        return id_list

    @classmethod
    def onehot_to_integer(cls, onehot):
        index = 0
        for i in onehot:
            if i == 1:
                return index
            index += 1
        raise Exception("Invalid One shot format")


def are_letters_sticky(x, y):
    # A sticks with the letter C ( and vice-versa) and the letter B sticks with the letter D ( and vice-versa)
    return (x == 'A' and y == 'C') or (x == 'C' and y == 'A') or (x == 'B' and y == 'D') or (x == 'D' and y == 'B')


def get_max_sticky(input_string):
    str_len = len(input_string)
    max_stick = 0
    for i in xrange(str_len/2):
        if not are_letters_sticky(input_string[i], input_string[str_len-i-1]):
            break
        max_stick += 1
    return max_stick


def get_label_tuple(index):
    id_list = LabelConverter.integer_to_onehot(index)
    return id_list, LABELS[index]

def get_label(input_string):
    max_stick = get_max_sticky(input_string)
    if max_stick == 0:
        return get_label_tuple(0)
    elif max_stick == 1 or max_stick == 2:
        return get_label_tuple(1)
    elif max_stick == 3 or max_stick == 4:
        return get_label_tuple(2)
    elif max_stick == 5 or max_stick == 6:
        return get_label_tuple(3)
    elif 7 <= max_stick < 20:
        return get_label_tuple(4)
    elif max_stick == 20:
        return get_label_tuple(5)


def test():
    print get_label("ABCBDCBDBCADBACBACDADACBDADBACBDACBBADC")

if __name__ == '__main__':
    test()