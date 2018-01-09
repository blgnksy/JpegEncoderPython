import math

amplitude = 3
size = int(math.log(abs(amplitude), 2)) + 1

print(size)
code = None
if size == 0:
    code = None
else:
    if amplitude > 0:
        to_code = pow(2, size - 1)
        code = bin((amplitude - to_code) | (1 << (size - 1)))
    else:
        to_code = pow(2, size - 1)
        code = bin(abs(amplitude - to_code) | (0 << (size - 1)))
print(code)
bytearrays=bytearray(code)

print(bytearrays)