from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

filename = 'iris.data'
names = ['seper-length', 'seper-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(filename, names = names)
print('数据维度：行 %s ，列 %s' % dataset.shape) #查看数据维度
print(dataset.head(15)) #查看前15行
print(dataset.describe()) #统计描述数据信息
print(dataset.groupby('class').size()) #分类分布情况

#数据可视化
dataset.hist()
pyplot.show() #直方图
dataset.plot(kind = 'box', subplots = True, layout = (2, 2), sharex = False, sharey = False)
pyplot.show() #箱线图
scatter_matrix(dataset)

#分离数据集
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = \
	train_test_split(X, Y, test_size = validation_size, random_state = seed)

#算法审查
models = {}
models['LR'] = LogisticRegression() #线性回归
models['LDA'] = LinearDiscriminantAnalysis() #线性判别分析
models['KNN'] = KNeighborsClassifier() #K近邻
models['CART'] = DecisionTreeClassifier() #分类与回归树
models['NB'] = GaussianNB() #贝叶斯分类器
models['SVM'] = SVC(); #支持向量机

#评估算法
results = []
for key in models :
	kflod = KFold(n_splits = 10, random_state = seed)
	cv_results = cross_val_score(models[key], X_train, Y_train, cv = kflod, scoring = 'accuracy')
	results.append(cv_results)
	print('%s: %f (%f)' % (key, cv_results.mean(), cv_results.std()))

#用箱线图比较算法
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(models.keys())
pyplot.show()

#使用评估数据集评估算法
svm = SVC()
svm.fit(X = X_train, y = Y_train)
predictions = svm.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))