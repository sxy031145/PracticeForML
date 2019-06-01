from pandas import read_csv
from pandas import set_option
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names = names)

peek = data.head(15)
print(peek) #前15行
print(data.shape) #维度
print(data.dtypes) #数据类型

#描述性统计
set_option('display.width', 80)
set_option('precision', 3)
print(data.describe())
# 数据数目 平均值 标准方差 最小值 下四分位数 中位数 上四分位数 最大值 

#数据分类分布
print(data.groupby('class').size())

#数据属性的相关性
set_option('display.width', 100)
set_option('precision', 2)
print(data.corr(method = 'pearson'))

#数据的分布分析
print(data.skew())
#正态分布的偏离情况