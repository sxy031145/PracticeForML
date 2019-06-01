from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler #调整尺度
from sklearn.preprocessing import StandardScaler #正态化
from sklearn.preprocessing import Normalizer #标准化
from sklearn.preprocessing import Binarizer #二值
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names = names)

#处理数据的一般流程：
#导入数据、按照I/O整理数据、格式化输入数据、总结显示数据的变化
#先将数据分为输入数据和输出结果
array = data.values
X = array[:, 0:8]
Y = array[:, 8]

#调整数据尺度
#将所有属性标准化，映射到[0,1]区间
transformer = MinMaxScaler(feature_range = (0, 1))
newX = transformer.fit_transform(X) #设定数据打印格式
set_printoptions(precision = 3)
print(newX)

#正态化数据
#处理符合正态分布的数据的手段
#输出结果均值0，方差1，可作为数据正态分布的算法的输入
transformer = StandardScaler().fit(X)
newX = transformer.transform(X)
set_printoptions(precision = 3)
print(newX)

#标准化数据
#将每一行的数据变成矢量距离为1
#对权重输入的神经网络和使用距离的K近邻有提升作用
transformer = Normalizer().fit(X)
newX = transformer.transform(X)
set_printoptions(precision = 3)
print(newX)

#二值数据
#大于阈值设为1，小于阈值设为0
transformer = Binarizer(threshold = 10.0).fit(X)
newX = transformer.transform(X)
set_printoptions(precision = 3)
print(newX)