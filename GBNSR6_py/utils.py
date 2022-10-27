from exceptions import PositionDecimalExceed
def get_numbers(line, limit = -1):
    new_digit = ''
    numbers = []
    is_neg = False
    for i in range(len(line)):
        c = line[i]
        if i < len(line) - 1 and c == '-' and line[i + 1].isdigit():
            is_neg = True
        if c.isdigit() or (c == '.' and new_digit != ''):
            new_digit += c
        if new_digit != '' and c != '.' and (i == len(line) - 1 or not c.isdigit()):
            new_num = float(new_digit)
            new_num = -new_num if is_neg else new_num
            numbers.append(new_num)
            if len(numbers) == limit:
                return numbers
            new_digit = ''
    return numbers

def generate_inpcrd_num(num):
    num_s = str(num)
    num_f = len(num_s.split('.')[1])

    if num_f > 7:
        raise PositionDecimalExceed("Number of acceptable decimals for floating part is 7 but position %d has more" % num)

    num_s += ''.join(['0' for i in range(7 - num_f)])
    num_d = len(num_s.split('.')[0])

    if num_d > 4:
        raise PositionDecimalExceed("Number of acceptable decimals for non-floating part is 4 but position %d has more" % num)

    num_s = ''.join([' ' for i in range(4 - num_d)]) + num_s
    return num_s

def add_zeros(num, max_len = 3):
    num_s = str(num)
    num_f = num_s.split('.')[1]

    num_f += ''.join(['0' for i in range(max_len - len(num_f))])

    return num_s.split('.')[0] + '.' + num_f
