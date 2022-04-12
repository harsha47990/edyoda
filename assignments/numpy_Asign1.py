import numpy as np

# 1. Create a null vector of size 10 but the fifth value which is 1.
ary = np.zeros(10)
ary[4] = 1
print(ary)

#2.Create a vector with values ranging from 10 to 49.
ary = np.arange(10,49)
print(ary)


#3. Create a 3x3 matrix with values ranging from 0 to 8
ary = np.arange(0,9)
ary = ary.reshape((3,3))
print(ary)

#4. Find indices of non-zero elements from [1,2,0,0,4,0]
ary = np.array([1,2,0,0,4,0])
print(ary.nonzero())

#5. Create a 10x10 array with random values and find the minimum and maximum values.
ary = np.random.random((10,10))
print('min',ary.min())
print('max',ary.max())


#6. Create a random vector of size 30 and find the mean value.
ary = np.random.random(30)
print('mean',ary.mean())
