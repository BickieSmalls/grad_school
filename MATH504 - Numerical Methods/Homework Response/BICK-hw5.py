import scipy
import numpy as np
import random

# problem 1

A = scipy.linalg.hilbert(5)

x = np.random.rand(5,1)

b = np.matmul(A,x)

epsilon = np.random.uniform(-1,1,(5,1)) / 1000
b_hat = b + epsilon

x_hat = np.linalg.solve(A, b_hat)


# Problem 2
A = np.array([
  [1,2,3],
  [4,5,6],
  [7,8,9]  
])

A_eig = np.linalg.eig(A)


# Problem 3
A = np.array([
    [0,1,1,1,0],
    [0,0,0,1,0],
    [0,0,0,0,1],
    [1,0,1,0,1],
    [0,0,0,1,0]
])

B = np.array([
    [0,1/3,1/3,1/3,0],
    [0,0,0,1,0],
    [0,0,0,0,1],
    [1/3,0,1/3,0,1/3],
    [0,0,0,1,0]
])

# then B transpose
B_T = B.transpose()

B_T_eig = np.linalg.eig(B_T)

B_T_evals = B_T_eig[0]
B_T_evecs = B_T_eig[1].transpose()

B_T_evecs[0]/B_T_evecs[0].sum(axis=0)