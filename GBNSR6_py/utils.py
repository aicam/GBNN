from exceptions import PositionDecimalExceed
def get_numbers(line):
    new_digit = ''
    numbers = []
    for c in line:
        if c.isdigit() or c == '.':
            new_digit += c
        if not c.isdigit() and c != '.' and new_digit != '':
            numbers.append(float(new_digit))
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