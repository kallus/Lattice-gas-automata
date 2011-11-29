import numpy as np
import c_module

matrix = np.zeros((3, 3), dtype=np.int64)
matrix[0, :] = [0, 1, 0]
matrix[1, :] = [0, 0, 0]
matrix[2, :] = [0, 0, 0]
matrix_temp = np.zeros((3, 3), dtype=np.int64)
matrix_temp[0, :] = [0, 0, 0]
matrix_temp[1, :] = [0, 0, 0]
matrix_temp[2, :] = [0, 0, 0]

print("T_temp=0")
print(str(matrix_temp))
print("T=0")
print(str(matrix))

c_module.c_module(matrix, matrix_temp)

print("T_temp=1")
print(str(matrix_temp))
print("T=1")
print(str(matrix))

c_module.c_module(matrix, matrix_temp)

print("T_temp=2")
print(str(matrix_temp))
print("T=2")
print(str(matrix))
