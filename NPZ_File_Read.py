import numpy as np
import os
import cv2

#specifiy the path
filename = 'D:/FYP/Phaedra_BE/imagedata_final.npz'

#load the file
data = np.load(filename)

arr1 = data["arr_0"]
arr2 = data["arr_1"]

print("Array 01 is ...")
print (arr1)
# print("Array 02 is ...")
# print (arr2)

#array size
array_size1 = arr1.size
array_size2 = arr2.size

data = np.load('imagedata2.npz')

# print("All the files in the npz file are :")
# print(data.files)

first_array_shape = data["arr_0"].shape
second_array_shape = data["arr_1"].shape

batch_size_1 = first_array_shape[0]
batch_size_2 = second_array_shape[0]

print('Batch size:', batch_size_1)
print('Batch size:', batch_size_2)

print("Array 01 size is ", array_size1)
print("Array 02 size is ", array_size2)