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