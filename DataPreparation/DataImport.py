#标准Python导入
from csv import reader
import numpy as np
filename = 'pima-indians-diabetes.data.csv'
with open(filename, 'rt') as raw_data :
	readers = reader(raw_data, delimiter = ',')
	x = list(readers)
	data = np.array(x).astype('float')
	print(data.shape)

#NumPy导入
from numpy import loadtxt
filename = 'pima-indians-diabetes.data.csv'
with open(filename, 'rt') as raw_data :
	data = loadtxt(raw_data, delimiter = ',')
	print(data.shape)

#Pandas导入
from pandas import read_csv
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names = names)
print(data.shape)
print(data)