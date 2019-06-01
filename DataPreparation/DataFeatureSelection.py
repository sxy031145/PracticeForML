from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest #单变量选定
from sklearn.feature_selection import chi2 #卡方检验
from sklearn.feature_selection import RFE #递归消除
from sklearn.linear_model import LogisticRegression #逻辑回归算法
from sklearn.decomposition import PCA #主要成分分析
from sklearn.ensemble import ExtraTreesClassifier #随机森林算法
import warnings
warnings.filterwarnings('ignore')
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names = names)

#特征选定能够选择有助于提高预测结果的特征数据
#有助于降低数据的拟合度、提高算法精度、减少训练时间
#主要包括单变量特征选定、递归特性消除、主要成分分析、特征的重要性
array = data.values
X = array[:, 0:8]
Y = array[:, 8]

#单变量特征选定
#用卡方检验的方式选出对结果影响最大的4个特征
test = SelectKBest(score_func = chi2, k = 4)
fit = test.fit(X, Y)
set_printoptions(precision = 3)
print(fit.scores_)
features = fit.transform(X)
print(features)

#递归特性消除
#选定一个基模型，每轮训练后消除若干权值系数的特征
#以逻辑回归算法为模型，选定影响最大的4个特征
model = LogisticRegression()
rfe = RFE(model, 4)
fit = rfe.fit(X, Y)
print("Number of features:")
print(fit.n_features_)
print("Selected features:")
print(fit.support_)
print("Features rank:")
print(fit.ranking_)

#主要成分分析
#进行数据降维，在聚类分析中利于对数据的简化分析和可视化
pca = PCA(n_components = 3)
fit = pca.fit(X)
set_printoptions(precision = 8)
print("Explained variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)

#特征重要性
#袋装决策树算法、随机森林算法、极端随机树算法计算特征重要性
#示例用随机森林算法计算特征重要性
model = ExtraTreesClassifier()
fit = model.fit(X, Y)
print(fit.feature_importances_)