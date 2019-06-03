from pandas import read_csv
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names = names)

#分类算法矩阵以逻辑回归为例
#包括：分类准确度、对数损失函数、AUC图、混淆矩阵、分类报告
array = data.values
X = array[:, 0:8]
Y = array[:, 8]

num_folds = 10
seed = 7
kfold = KFold(n_splits = num_folds, random_state = seed)
model = LogisticRegression()

#分类准确度
#自动分类正确的样本数除以所有的样本数得出的结果
#但是准确度高并不代表算法一定好，例如数据分布不均衡的情况
result = cross_val_score(model, X, Y, cv = kfold)
print("KFold Ans: %.3f (%.3f)" % (result.mean(), result.std()))

#对数损失函数
#max{F(y, f(x))} -> min{-F(y, f(x))}
#对数损失函数越小模型就越好，而且使损失函数尽量是一个凸函数便于计算
scoring = 'neg_log_loss'
result = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)
print("Logless: %.3f (%.3f)" % (result.mean(), result.std()))

#AUC图
#定义敏感性为真正类率，识别出的正实例占所有正实例的比例
#定义特异性为真负类率，识别出的负实例占所有负实例的比例
#定义工作特性曲线ROC为敏感性为纵坐标，1-特异性(负正类率)为横坐标的图线
#AUC为ROC的积分，值越大准确性越高
scoring = 'roc_auc'
result = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)
print("AUC: %.3f (%.3f)" % (result.mean(), result.std()))

test_size = 0.33
seed = 4
X_train, X_test, Y_traing, Y_test = train_test_split(X, Y, test_size = test_size, random_state = seed)
model = LogisticRegression()
model.fit(X_train, Y_traing)
predicted = model.predict(X_test)
#混淆矩阵
#用于比较分类结果和实际测得值
#第i列第j行的位置是预测为类i实际为类j的数目
matrix = confusion_matrix(Y_test, predicted)
classes = {'0', '1'}
dataframe = pd.DataFrame(data = matrix, index = classes, columns = classes)
print("Confusion Matirx:")
print(dataframe)

#分类报告
#给出精确率P：检测为真的项目中实际为真的项目所占的比例
#给出召回率R：检测正确的项目中为真的项目的比例
#给出F1值：调和均值(P+R)/2
#给出样本数目
report = classification_report(Y_test, predicted)
print("Report:")
print(report)