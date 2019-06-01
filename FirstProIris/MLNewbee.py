#NumPy
import numpy as np
array = np.array([1, 4, 7, 6])
print(array)
print(array.shape)
array = np.array([[2, 5, 4], [7, 6, 5], [2, 1, 9]])
print(array)
print(array.shape)
print(array[1])
print(array[2][1])
print(array[-2])
print(array[-1, 0])
array1 = np.array([[2, 5, 4], [2, 8, 6], [9, 5, 1]])
print(array + array1)
print(array1 * array)

#Matplotlib
import matplotlib.pyplot as plt
plt.plot(array)
plt.xlabel('xThis')
plt.ylabel('yThis')
plt.show()
plt.scatter(array, array1)
plt.xlabel('xThis')
plt.ylabel('yThis')
plt.show()

#Pandas
import pandas as pd
Array = np.array([4, 7, 3])
Index = np.array(['r', 'b', 'k'])
Ans = pd.Series(Array, index = Index)
print(Ans)
print(Ans[0])
print(Ans['b'])
rowindex = ['row1', 'row2', 'row3']
colname = ['col1', 'col2', 'col3']
Frame = pd.DataFrame(data = array, index = rowindex, columns = colname)
print(Frame)
print(Frame['col2'])