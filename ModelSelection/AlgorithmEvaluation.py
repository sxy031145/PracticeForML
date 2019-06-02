from pandas import read_csv
from sklearn.model_selection import train_test_split #分离数据集
from sklearn.model_selection import KFold #K折
from sklearn.model_selection import LeaveOneOut #弃一
from sklearn.model_selection import ShuffleSplit #重复随机分离
from sklearn.model_selection import cross_val_score #交叉验证
from sklearn.linear_model import LogisticRegression #逻辑回归模型
import warnings
warnings.filterwarnings('ignore')
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names = names)

#将数据分为训练数据集和评估数据集，先训练模型再评估算法，4种方法:
#分离训练数据集和评估数据集、K折交叉验证分离、弃一交叉验证分离和重复随机评估、训练数据集分离
array = data.values
X = array[:, 0:8]
Y = array[:, 8]

#当不知道如何选择方法的时候选K折就对了
#当不知道K值取多少的时候设为10就对了
#我diao你妈的

#直接分离，三分之二训练集，三分之一评估集
#示例评估逻辑回归模型
test_size = 0.33
seed = 4 #指定粒度确保每次执行程序得到相同的结果
X_train, X_test, Y_traing, Y_test = train_test_split(X, Y, test_size = test_size, random_state = seed)
model = LogisticRegression()
model.fit(X_train, Y_traing)
result = model.score(X_test, Y_test)
print("Split Ans: %.3f%%" % (result*100))

#K折交叉验证分离
#数据K等分，每次将一组作为评估集，剩余的作为训练集，将K个模型的结果取均值
#K通常取3、5、10
num_folds = 10
seed = 7
kfold = KFold(n_splits = num_folds, random_state = seed)
model = LogisticRegression()
result = cross_val_score(model, X, Y, cv = kfold)
print("KFold Ans: %.3f%% (%.3f%%)" % (result.mean()*100, result.std()*100)) #得分与方差

#弃一交叉验证分离
#实现与K折基本相同，弃一交叉是每个样本单独作为验证集，其余作为训练集
#与K折相比，最接近原始样本的分布，结果可靠，没有随机因素影响数据，确保结果可被复制
#但是这个方法的速度很慢啊我diao你妈的
#而且结果还不如K折啊真就白给呗
loocv = LeaveOneOut()
model = LogisticRegression()
result = cross_val_score(model, X, Y, cv = loocv)
print("LeaveOne Ans: %.3f%% (%.3f%%)" % (result.mean()*100, result.std()*100))

#重复随机分离评估数据集与训练数据集
#就是字面意思一个非常无赖的方法
#随机分离数据集然后重复若干次
#非酋不配做机器学习我diao你妈的
n_splits = 10
test_size = 0.33
seed = 7
kfold = ShuffleSplit(n_splits = n_splits, test_size = test_size, random_state = seed)
model = LogisticRegression()
result = cross_val_score(model, X, Y, cv = kfold)
print("Shuffle Ans: %.3f%% (%.3f%%)" % (result.mean()*100, result.std()*100))