from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names = names)
#用sklearn审查六种机器学习的分类算法，2线性4非线性
#算法审查：通过实验判断出哪些算法最有效，再进一步选择
#线性：逻辑回归、线性判别分析
#非线性：K近邻、贝叶斯分类器、分类与回归树、支持向量机
#用10折交叉验证评估准确度，用平均准确度评估得分
array = data.values
X = array[:, 0:8]
Y = array[:, 8]
num_folds = 10
seed = 7
kfold = KFold(n_splits = num_folds, random_state = seed)

#逻辑回归：拟合一个逻辑函数
model = LogisticRegression()
result = cross_val_score(model, X, Y, cv = kfold)
print(result.mean())

#线性判别分析(LDA)
#将高维的模式样本投影到最佳鉴别矢量空间
model = LinearDiscriminantAnalysis()
result = cross_val_score(model, X, Y, cv = kfold)
print(result.mean())

#K近邻：如果一个样本的k个最相似的样本中大多数属于某一个类别
#则该样本也属于这个类别
model = KNeighborsClassifier()
result = cross_val_score(model, X, Y, cv = kfold)
print(result.mean())

#贝叶斯分类器
#选择具有最大后验概率的类作为该对象所属的类
model = GaussianNB()
result = cross_val_score(model, X, Y, cv = kfold)
print(result.mean())

#分类与回归树(CART)
#基于训练数据集构建决策树，用验证数据集进行剪枝
model = DecisionTreeClassifier()
result = cross_val_score(model, X, Y, cv = kfold)
print(result.mean())

#支持向量机(SVM)
model = SVC()
result = cross_val_score(model, X, Y, cv = kfold)
print(result.mean())