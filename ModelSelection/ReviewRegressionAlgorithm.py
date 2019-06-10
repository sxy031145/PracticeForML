from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PRTATIO', 'B', 'LSTAT', 'MEDV']
data = read_csv(filename, names = names)
#10折交叉验证分离数据，用均方误差评估算法模型
array = data.values
X = array[:, 0:13]
Y = array[:, 13]
n_splits = 10
seed = 7
kfold = KFold(n_splits = n_splits, random_state = seed)

#线性回归算法
model = LinearRegression()
scoring = 'neg_mean_squared_error'
result = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)
print("Linear Regression: %.3f" % result.mean())

#岭回归算法（改良的最小二乘估计）
model = Ridge()
scoring = 'neg_mean_squared_error'
result = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)
print("Ridge Regression: %.3f" % result.mean())

#套索回归算法
#使用的惩罚值是绝对值而不是平方
model = Lasso()
scoring = 'neg_mean_squared_error'
result = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)
print("Lasso Regression: %.3f" % result.mean())

#弹性网络回归算法
#套索回归和岭回归的混合体，随机挑选其中一个
model = ElasticNet()
scoring = 'neg_mean_squared_error'
result = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)
print("ElasticNet Regression: %.3f" % result.mean())

#K近邻算法
#默认闵式距离，可以指定曼哈顿距离
model = KNeighborsRegressor()
scoring = 'neg_mean_squared_error'
result = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)
print("KNeighbors Regression: %.3f" % result.mean())

#分类与回归树
model = DecisionTreeRegressor()
scoring = 'neg_mean_squared_error'
result = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)
print("CART: %.3f" % result.mean())

#支持向量机
model = SVR()
scoring = 'neg_mean_squared_error'
result = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)
print("SVM: %.3f" % result.mean())