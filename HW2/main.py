import numpy as np 
c1 = np.array([[5, 3], [3, 5], [3, 4], [4, 5], [4, 7], [5, 6]]) 
c2 = np.array([[9, 10], [7, 7], [8, 5], [8, 8], [7, 2], [10, 8]])

def getMean(c):
    return np.array([np.mean(c[:, 0]), np.mean(c[:, 1])])

def WithinMatrix(c, m):
    return ( np.matmul(  np.transpose(c-m), c-m ) )

mean1 = getMean(c1)
mean2 = getMean(c2)
mean = (mean1 + mean2) / 2

sw1 = WithinMatrix(c1, mean1)
sw2 = WithinMatrix(c2, mean2)
sw = sw1 + sw2

sb1 = np.matmul( (mean1 - mean).reshape([2,1]), np.transpose((mean1 - mean).reshape([2,1])) )
sb2 = np.matmul( (mean2 - mean).reshape([2,1]), np.transpose((mean2 - mean).reshape([2,1])) )
sb = sb1 + sb2

eigval, eigvec = np.linalg.eig( np.matmul(np.linalg.inv(sw), sb) )

print("eigenvalues:\n",eigval,"\n")
print("eigenvectors:\n",eigvec)