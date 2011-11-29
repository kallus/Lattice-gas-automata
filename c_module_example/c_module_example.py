import numpy as np
import c_module

matrix = np.zeros((3, 3), dtype=np.int64)
matrix[0, :] = [1, 2, 3]
matrix[1, :] = [4, 5, 6]
matrix[2, :] = [7, 8, 9]
py_list = []

print("Before C:")
print(str(py_list))
print(str(matrix))

c_module.c_module(2, py_list, 2.3, matrix)

print("After C:")
print(str(py_list))
print(str(matrix))
