from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PRTATIO', 'B', 'LSTAT', 'MEDV']
data = read_csv(filename, names = names)
#回归算法矩阵包括：
#平均绝对误差、均方误差、决定系数
array = data.values
X = array[:, 0:13]
Y = array[:, 13]
n_splits = 10
seed = 7
kfold = KFold(n_splits = n_splits, random_state = seed)
model = LinearRegression()

#平均绝对误差：所有观测值与算数平均值之差的绝对值的平均值
scoring = 'neg_mean_absolute_error'
result = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)
print("MAE: %.3f (%.3f)" % (result.mean(), result.std()))

#均方误差
scoring = 'neg_mean_squared_error'
result = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)
print("MSE: %.3f (%.3f)" % (result.mean(), result.std()))

#决定系数：因变量的全部变异能通过回归关系被自变量解释的比例
scoring = 'r2'
result = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)
print("R2: %.3f (%.3f)" % (result.mean(), result.std()))